"""Stage a deployment spec into a tritonserver-mountable model repository.

Usage:
  python deploy.py <deployment.yaml> [--model-repository model_repository]

Given a deployment spec such as:

    model_name: example
    task: text-classification-hf
    weight_dir: /path/to/weights
    venv: vllm-v0_26_0.tar.gz
    version: 1                     # optional, default 1
    config:
      max_batch_size: 256
      param_tokenizer_max_length: '1024'

this produces:

    <model_repository>/<model_name>/
      config.pbtxt          rendered from tasks/<task>/config.pbtxt.jinja
      venv.tar.gz           venv-builder/envs/<venv>, placed per EXECUTION_ENV_PATH
      <version>/
        model.py, ...       copied from tasks/<task>/src/
        model/              copied from <weight_dir>

which can be mounted directly:

    docker run -v $(pwd)/model_repository:/models ... tritonserver --model-repository=/models
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO_ROOT / "tasks"
ENVS_DIR = REPO_ROOT / "venv-builder" / "envs"
DEFAULT_MODEL_REPOSITORY = REPO_ROOT / "model_repository"

# Weights are loaded by every task's model.py as `<version_dir>/model`
WEIGHT_DIR_NAME = "model"
IGNORE_PATTERNS = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")

# parameters: { key: "EXECUTION_ENV_PATH" value: { string_value: "..." } }
EXECUTION_ENV_RE = re.compile(
    r"""key:\s*"EXECUTION_ENV_PATH"\s*,?\s*value:\s*\{\s*string_value:\s*"([^"]+)"\s*\}""",
    re.MULTILINE,
)


class DeployError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[deploy] {msg}", file=sys.stderr)


# --------------------------------------------------------------------------- spec


def load_spec(path: Path) -> dict[str, Any]:
    with open(path) as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict):
        raise DeployError(f"{path}: deployment spec must be a YAML mapping")

    for key in ("model_name", "task", "weight_dir"):
        if not spec.get(key):
            raise DeployError(f"{path}: missing required field '{key}'")

    known = {"model_name", "task", "weight_dir", "venv", "version", "config"}
    unknown = set(spec) - known
    if unknown:
        raise DeployError(f"{path}: unknown field(s): {', '.join(sorted(unknown))}")

    config = spec.get("config") or {}
    if not isinstance(config, dict):
        raise DeployError(f"{path}: 'config' must be a mapping")
    spec["config"] = config
    return spec


def resolve_path(value: str, base: Path) -> Path:
    """Resolve a spec path: absolute as-is, relative against the repo root."""
    path = Path(os.path.expanduser(str(value)))
    return path if path.is_absolute() else (base / path)


def resolve_venv(venv: str) -> Path:
    """Locate a venv tarball by name under venv-builder/envs, or by path."""
    candidate = resolve_path(venv, ENVS_DIR)
    names = [candidate]
    # spec may name the venv without (or with a partial) extension
    if not candidate.exists():
        names += [candidate.with_name(candidate.name + suffix) for suffix in (".gz", ".tar.gz")]
    for name in names:
        if name.is_file():
            return name
    raise DeployError(f"venv not found: tried {', '.join(str(n) for n in names)}")


# --------------------------------------------------------------------------- copy


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, ignore=IGNORE_PATTERNS)


# --------------------------------------------------------------------------- render


def build_context(spec: dict[str, Any]) -> dict[str, Any]:
    """Template context: config keys as written, plus upper/lower-case aliases.

    Task templates are inconsistent (`max_batch_size` vs `MAX_BATCH_SIZE`), so
    each key is exposed in both casings unless that would clobber an explicit one.
    """
    context: dict[str, Any] = {"model_name": spec["model_name"]}
    config: dict[str, Any] = spec["config"]
    for key, value in config.items():
        for alias in (str(key), str(key).upper(), str(key).lower()):
            if alias in config and alias != key:
                continue  # explicitly set elsewhere, don't overwrite
            context[alias] = value
    return context


def render_config(template_path: Path, output_path: Path, context: dict[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    rendered = env.get_template(template_path.name).render(**context)
    output_path.write_text(rendered)
    return rendered


def venv_destination(rendered_config: str, model_dir: Path) -> tuple[Path | None, str | None]:
    """Where the venv tarball must land for this config.

    Returns (destination, note). A destination of None means the config points
    at a path outside the model directory (e.g. /tmp/venv.tar.gz), which has to
    be provided as a mount instead.
    """
    match = EXECUTION_ENV_RE.search(rendered_config)
    if match is None:
        return model_dir / "venv.tar.gz", (
            "config.pbtxt has no EXECUTION_ENV_PATH parameter; the staged venv will be "
            "ignored by tritonserver unless the task template is updated"
        )

    value = match.group(1)
    # Triton expands $$TRITON_MODEL_DIRECTORY to the model directory
    if "$$TRITON_MODEL_DIRECTORY" in value:
        relative = value.replace("$$TRITON_MODEL_DIRECTORY", "").lstrip("/")
        return model_dir / relative, None
    return None, (
        f"EXECUTION_ENV_PATH is an absolute path ({value}); mount the venv there yourself, "
        f'e.g. -v <venv>:{value}:ro'
    )


# --------------------------------------------------------------------------- main


def deploy(spec_path: Path, model_repository: Path, force: bool) -> Path:
    spec = load_spec(spec_path)

    model_name = str(spec["model_name"])
    task = str(spec["task"])
    version = str(spec.get("version", 1))

    task_dir = TASKS_DIR / task
    src_dir = task_dir / "src"
    template_path = task_dir / "config.pbtxt.jinja"
    if not src_dir.is_dir():
        raise DeployError(f"task source not found: {src_dir}")
    if not template_path.is_file():
        raise DeployError(f"task config template not found: {template_path}")
    if not (src_dir / "model.py").is_file():
        raise DeployError(f"task source has no model.py: {src_dir}")

    weight_dir = resolve_path(spec["weight_dir"], REPO_ROOT)
    if not weight_dir.is_dir():
        raise DeployError(f"weight_dir not found: {weight_dir}")

    venv_path = resolve_venv(str(spec["venv"])) if spec.get("venv") else None

    model_dir = model_repository / model_name
    if model_dir.exists():
        if not force:
            raise DeployError(f"{model_dir} already exists (use --force to replace)")
        log(f"removing existing {model_dir}")
        try:
            shutil.rmtree(model_dir)
        except PermissionError as exc:
            raise DeployError(
                f"cannot remove {model_dir}: {exc}. It likely contains files written by a "
                f"container running as root; remove them with:\n"
                f"    docker run --rm -v {model_repository}:/repo <tritonserver-image> "
                f"rm -rf /repo/{model_name}"
            ) from exc

    version_dir = model_dir / version
    version_dir.mkdir(parents=True)

    # 1. task code -> <model>/<version>/
    log(f"code    {src_dir} -> {version_dir}")
    for entry in sorted(src_dir.iterdir()):
        if entry.name in ("__pycache__",) or entry.suffix == ".pyc":
            continue
        if entry.is_dir():
            copy_tree(entry, version_dir / entry.name)
        else:
            shutil.copy2(entry, version_dir / entry.name)

    # 2. config.pbtxt.jinja -> <model>/config.pbtxt
    config_path = model_dir / "config.pbtxt"
    rendered = render_config(template_path, config_path, build_context(spec))
    log(f"config  {template_path} -> {config_path}")

    # 3. weights -> <model>/<version>/model/
    weight_dst = version_dir / WEIGHT_DIR_NAME
    log(f"weights {weight_dir} -> {weight_dst}")
    copy_tree(weight_dir, weight_dst)

    # 4. venv tarball -> wherever EXECUTION_ENV_PATH points
    if venv_path is not None:
        destination, note = venv_destination(rendered, model_dir)
        if note:
            log(f"WARNING {note}")
        if destination is not None:
            log(f"venv    {venv_path} -> {destination}")
            copy_file(venv_path, destination)
    else:
        _, note = venv_destination(rendered, model_dir)
        if note is None:
            log("WARNING config.pbtxt expects a venv (EXECUTION_ENV_PATH) but spec has no 'venv'")

    return model_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", nargs="+", type=Path, help="deployment yaml file(s)")
    parser.add_argument("--model-repository", type=Path, default=DEFAULT_MODEL_REPOSITORY,
                        help=f"output model repository (default: {DEFAULT_MODEL_REPOSITORY})")
    parser.add_argument("--force", action="store_true", help="replace an existing model directory")
    args = parser.parse_args()

    model_repository: Path = args.model_repository.resolve()
    model_repository.mkdir(parents=True, exist_ok=True)

    deployed: list[Path] = []
    try:
        for spec_path in args.spec:
            log(f"=== {spec_path}")
            deployed.append(deploy(spec_path, model_repository, args.force))
    except DeployError as exc:
        print(f"[deploy] ERROR {exc}", file=sys.stderr)
        raise SystemExit(1)

    log("done")
    names = " ".join(d.name for d in deployed)
    load_flags = " ".join(f"--load-model {d.name}" for d in deployed)
    print(f"""
model repository: {model_repository}

serve with:
    ./serve.sh {names}

or directly (read-only mount, so the container cannot write into the repository):
    docker run --rm --gpus all \\
        -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \\
        -e PYTHONDONTWRITEBYTECODE=1 \\
        -p 8000:8000 -p 8001:8001 -p 8002:8002 \\
        -v {model_repository}:/models:ro \\
        nvcr.io/nvidia/tritonserver:26.06-py3 \\
        tritonserver --model-repository=/models \\
        --model-control-mode=explicit {load_flags}
""")


if __name__ == "__main__":
    main()
