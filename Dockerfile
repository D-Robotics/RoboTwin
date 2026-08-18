ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ARG UV_VERSION=0.9.1
ARG CUDA_VARIANT=cu124
ARG EXTRA_LD_LIB_PATH=""

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs curl ca-certificates \
    build-essential cmake ninja-build \
    libegl1-mesa-dev libglew-dev libglfw3-dev \
    libgles2-mesa-dev libglib2.0-0 libsm6 libxrender1 libxext6 \
    libosmesa6-dev ffmpeg \
    libvulkan1 vim \
    $([ "$CUDA_VARIANT" = "cu124" ] && echo "libgl1-mesa-glx python3.11 python3.11-venv python3.11-dev" || echo "libgl1 libglx-mesa0") \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/

WORKDIR /workspace/RoboTwin

COPY policy/pi0/pyproject.toml policy/pi0/uv.lock policy/pi0/.python-version policy/pi0/LICENSE policy/pi0/
COPY policy/pi0/packages policy/pi0/packages
COPY policy/pi0/src policy/pi0/src
COPY policy/pi0/vendor/lerobot policy/pi0/vendor/lerobot

RUN if [ "$CUDA_VARIANT" = "cu124" ]; then \
        cd policy/pi0 && \
        sed -i 's|lerobot = { git = "https://github.com/huggingface/lerobot", rev = "a445d9c9da6bea99a8972daa4fe1fdd053d711d2" }|lerobot = { path = "vendor/lerobot" }|' pyproject.toml && \
        GIT_LFS_SKIP_SMUDGE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never uv sync --frozen --python python3.11 && \
        cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/; \
    else \
        uv python install 3.11 && \
        cd policy/pi0 && \
        sed -i 's|lerobot = { git = "https://github.com/huggingface/lerobot", rev = "a445d9c9da6bea99a8972daa4fe1fdd053d711d2" }|lerobot = { path = "vendor/lerobot" }|' pyproject.toml && \
        GIT_LFS_SKIP_SMUDGE=1 UV_LINK_MODE=copy uv sync --frozen --python 3.11 --no-install-package torch --no-install-package torchvision && \
        . .venv/bin/activate && \
        python -VV && \
        python -c "import sys; assert sys.version_info >= (3, 11, 13), 'must use >=3.11.13: ' + sys.version" && \
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 && \
        cp -r src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/; \
    fi

COPY envs/curobo envs/curobo
RUN test -n "$(ls envs/curobo/src/curobo/curobolib/*.so 2>/dev/null)" || \
    (echo "ERROR: missing curobo .so, run bash script/prepare-docker-build.sh first" && exit 1)
RUN cd policy/pi0 && . .venv/bin/activate && \
    cd ../../envs/curobo && \
    sed -i 's/ext_modules=ext_modules/ext_modules=[]/' setup.py && \
    sed -i 's/cmdclass={"build_ext": BuildExtension}/cmdclass={}/' setup.py && \
    pip install -e . --no-build-isolation && \
    pip install warp-lang==1.8.0

# cu128 冒烟测试：curobo 导入触发 torch.jit.script，3.11.0rc1 会段错误
RUN if [ "$CUDA_VARIANT" = "cu128" ]; then \
        cd policy/pi0 && . .venv/bin/activate && \
        python -c "import curobo.graph.graph_base; import curobo.types.math; print('curobo import OK')"; \
    fi

ENV VIRTUAL_ENV=/workspace/RoboTwin/policy/pi0/.venv \
    PATH="/workspace/RoboTwin/policy/pi0/.venv/bin:${PATH}" \
    PYTHONPATH=/workspace/RoboTwin:/workspace/RoboTwin/policy/pi0/src \
    NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute \
    LD_LIBRARY_PATH=${EXTRA_LD_LIB_PATH}/usr/lib/x86_64-linux-gnu:/workspace/RoboTwin/policy/pi0/.venv/lib/python3.11/site-packages/torch/lib \
    CUROBO_TORCH_COMPILE_DISABLE=0 \
    VK_ICD_FILENAMES=/workspace/RoboTwin/policy/pi0/.venv/lib/python3.11/site-packages/sapien/vulkan_library/nvidia_icd.json

COPY . .

RUN sed -i 's|path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})|path = Path("/workspace/RoboTwin/policy/pi0/src/openpi/shared/big_vision/paligemma_tokenizer.model")|' \
    policy/pi0/src/openpi/models/tokenizer.py

CMD ["/bin/bash"]
