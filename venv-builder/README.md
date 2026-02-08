# tritonserver python venv builder

## Usage
Build and export a packed environment from a requirements file under `requirements/`:

```bash
./build-venv.sh <requirements_name>
```

Example:

```bash
./build-venv.sh text-clf-vllm-v1
```

This reads `requirements/<requirements_name>.txt`, creates `venv.tar.gz` inside
the container, and writes the artifact to:

```bash
envs/<requirements_name>.tar.gz
```
