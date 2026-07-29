# tritonserver-serving
Model serving test with tritonserver

## Repository Structure
- [tasks](./tasks): model inference code (`model.py`) & config (`config.pbtxt`) template
- [venv-builder](./venv-builder/): conda-pack venv tar file builder (to be injected as runtime python env)

## Supported Tasks
Supported model inference tasks

| Task | Description |
| --- | --- |
| [text-classification-hf](./tasks/text-classification-hf) | Transformers `AutoModelForSequenceClassification` based text classification model |
| [text-classification-vllm](./tasks/text-classification-vllm) | vLLM pooling engine `classify` task based text classification model ([reference](https://docs.vllm.ai/en/stable/models/pooling_models/#llmclassify)) |
| [span-detection-hf](./tasks/span-detection-hf) | Transformers `AutoModelForTokenClassification` based text BIO span tagging model |

## Usage
Describe a deployment in [deployments](./deployments), stage it into `model_repository/`, then serve it.

```yaml
# deployments/example.yaml
model_name: example                 # triton model name & directory name
task: text-classification-hf        # tasks/<task>
weight_dir: model/weights/clf-qwen3 # absolute, or relative to the repo root
venv: vllm-v0_26_0.tar.gz           # venv-builder/envs/<venv>, or a path
version: 1                          # optional (default 1)
config:                             # config.pbtxt.jinja template variables
  max_batch_size: 256
```

```bash
./deploy.sh deployments/example.yaml [--force]
./serve.sh deployments/example.yaml   # or: ./serve.sh example  /  ./serve.sh (all staged models)
```

`deploy.sh` uses whatever python environment is currently active (requires `pyyaml` and
`jinja2`), so activate your env first.

`deploy.sh` produces a directly mountable repository:

```
model_repository/example/
├── config.pbtxt   # rendered from tasks/<task>/config.pbtxt.jinja
├── venv.tar.gz    # placed at the path EXECUTION_ENV_PATH points to
└── 1/
    ├── model.py   # copied from tasks/<task>/src/
    └── model/     # copied from weight_dir
```

Weights and the venv tarball are copied, so the staged repository is fully self-contained and
independent of the source files.

`serve.sh` mounts `model_repository/` at `/models` **read-only**. tritonserver runs as root in the
container, so a writable mount leaves root-owned files (`__pycache__`) in the host tree; the
python backend unpacks `EXECUTION_ENV_PATH` into `/tmp` inside the container and never needs to
write to `/models`. `PYTHONDONTWRITEBYTECODE=1` is set as well.

Both scripts read settings from `.env` (see [.env.example](./.env.example)); environment
variables override it:

| Variable | Default | Used by |
| --- | --- | --- |
| `TRITONSERVER_IMAGE` | `nvcr.io/nvidia/tritonserver:26.06-py3` | serve.sh |
| `MODEL_REPOSITORY` | `<repo>/model_repository` | both |
| `HTTP_PORT` / `GRPC_PORT` / `METRICS_PORT` | `8000` / `8001` / `8002` | serve.sh |
| `GPUS` / `SHM_SIZE` | `all` / `8g` | serve.sh |
