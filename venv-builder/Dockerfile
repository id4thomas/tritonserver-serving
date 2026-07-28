FROM nvcr.io/nvidia/tritonserver:26.06-py3-sdk

# Prevent user-site packages from leaking into the environment
ENV PYTHONNOUSERSITE=True
ENV CUDA_HOME="/usr/local/cuda"

# Install Miniconda (detect arch for x86_64 vs aarch64)
RUN ARCH=$(uname -m) && \
    wget -qO /tmp/miniconda.sh "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCH}.sh" && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH="/opt/conda/bin:${PATH}"

# Create the conda env with Python 3.12 (must match Triton's python_backend stub)
# and install conda-pack. Use --override-channels to avoid Anaconda TOS prompt.
RUN conda create -y -n triton_env --override-channels -c conda-forge python=3.12 && \
    conda install -y -n triton_env --override-channels -c conda-forge conda-pack

# At runtime: a requirements folder is mounted into /tmp/requirements/,
# then files are installed in sorted order and the environment is packed.
CMD for f in $(ls /tmp/requirements/*.txt | sort); do \
      echo "==> Installing $f ..." && \
      conda run -n triton_env pip install --no-cache-dir -r "$f" || exit 1; \
    done && \
    conda run -n triton_env conda-pack -n triton_env -o /tmp/venv.tar.gz \
      --ignore-editable-packages --ignore-missing-files