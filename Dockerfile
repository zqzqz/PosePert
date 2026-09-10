# PosePert reproduction environment.
#
#   docker build -t posepert .
#   docker run --gpus all -it \
#       -v $PWD/data:/workspace/PosePert/data \
#       -v $PWD/models:/workspace/PosePert/models \
#       posepert bash
#
# data/ and models/ are mounted rather than baked in: they are ~7.5 GB of release
# archives plus the public datasets, and mounting keeps them shared between runs.
# See docs/INSTALL.md.
#
# Build arguments:
#   WITH_GRIP=1     clone AdvTrajectoryPrediction (~320 MB). Required for every
#                   scenario-level experiment, since GRIP++ itself lives there.
#   WITH_V2XREAL=0  clone and patch the V2X-Real API (~1.3 GB). Only needed for
#                   the V2X-Real setting; off by default to keep the image small.
#
# The CARLA closed-loop study is NOT covered by this image: it needs a different
# Python/PyTorch generation and a running CARLA server. See carla_demo/SETUP.md.

FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ARG WITH_GRIP=1
ARG WITH_V2XREAL=0

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates unzip \
    build-essential cmake \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

RUN conda install -y python=3.7.11 && conda clean -ya

COPY environment.yml /tmp/environment.yml
RUN conda env update -n base --file /tmp/environment.yml && conda clean -ya

RUN conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 \
    pytorch-cuda=11.6 -c pytorch -c nvidia && conda clean -ya
RUN pip install --no-cache-dir spconv-cu116

WORKDIR /workspace/PosePert
COPY . /workspace/PosePert

# OpenCOOD and the CUDA IoU operator. These two compile steps are the ones most
# likely to fail in a hand-built environment, which is the main reason to use this
# image at all.
RUN cd third_party/OpenCOOD \
    && pip install --no-cache-dir -e . \
    && python opencood/utils/setup.py build_ext --inplace \
    && python opencood/pcdet_utils/setup.py build_ext --inplace
RUN cd mvp/perception/cuda_op && python setup.py install

# GRIP++ lives inside AdvTrajectoryPrediction; scenario experiments import it.
RUN if [ "$WITH_GRIP" = "1" ]; then \
        git clone --depth 1 https://github.com/zqzqz/AdvTrajectoryPrediction.git \
            third_party/AdvTrajectoryPrediction ; \
    fi

# V2X-Real, with the ego-id patch applied. Upstream uses ego_id = -1 as a sentinel
# and asserts on it, but V2X-Real numbers vehicle agents negatively, so a scene
# whose ego is vehicle -1 trips the assertion.
RUN if [ "$WITH_V2XREAL" = "1" ]; then \
        git clone --depth 1 https://github.com/ucla-mobility/V2X-Real.git \
            third_party/V2X-Real \
        && cd third_party/V2X-Real \
        && pip install --no-cache-dir -e . \
        && git apply ../patches/v2x-real-ego-id.patch \
        && cd /workspace/PosePert \
        && cp third_party/patches/point_pillar_intermediate_fusion_*.yaml \
              third_party/V2X-Real/opencood/hypes_yaml/ ; \
    fi

CMD ["bash", "-lc", "python scripts/check_artifact.py; echo; echo 'Mount data/ and models/, then see docs/EXPERIMENTS.md.'"]
