"""Render a Triton config.pbtxt from a Jinja template.

Usage:
  python render-config.py <template.jinja> <output.pbtxt> [--vars <JSON|@file.json>]

Template variables come from environment variables, then are overlaid by --vars
(JSON object literal, or @path to a JSON file). --vars wins on conflicts.
"""
import argparse
import json
import os
import sys

from jinja2 import Environment, FileSystemLoader, StrictUndefined


def load_vars(spec: str) -> dict:
    if spec.startswith("@"):
        with open(spec[1:]) as f:
            data = json.load(f)
    else:
        data = json.loads(spec)
    if not isinstance(data, dict):
        raise ValueError("--vars must be a JSON object")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template")
    parser.add_argument("output")
    parser.add_argument("--vars", dest="vars_spec", default=None,
                        help="JSON object literal, or @path to JSON file")
    args = parser.parse_args()

    context: dict = dict(os.environ)
    if args.vars_spec:
        context.update(load_vars(args.vars_spec))

    template_dir = os.path.dirname(os.path.abspath(args.template)) or "."
    env = Environment(
        loader=FileSystemLoader(template_dir),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    rendered = env.get_template(os.path.basename(args.template)).render(**context)

    with open(args.output, "w") as f:
        f.write(rendered)
    print(f"rendered {args.template} -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
