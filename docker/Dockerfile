FROM docker-registry.qualcomm.com/library/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.9

RUN apt-get clean
RUN apt-get update --fix-missing

RUN apt-get install -y --no-install-recommends \
    software-properties-common

# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

RUN apt-get update --fix-missing
RUN apt-get install -y --no-install-recommends \
    curl sudo vim \
    autoconf automake libtool make g++ unzip cmake wget clang \
    git git-lfs openssh-server ffmpeg libxtst6 mysql-server libmysqlclient-dev libopenmpi-dev 

# install RDMA required packages
RUN wget -q --no-check-certificate https://content.mellanox.com/ofed/MLNX_OFED-23.10-2.1.3.1/MLNX_OFED_LINUX-23.10-2.1.3.1-ubuntu20.04-x86_64.tgz && \
    tar -xvf ./MLNX_OFED_LINUX-23.10-2.1.3.1-ubuntu20.04-x86_64.tgz -C /opt/
RUN apt-get update --fix-missing
RUN echo 'deb file:/opt/MLNX_OFED_LINUX-23.10-2.1.3.1-ubuntu20.04-x86_64/DEBS ./ ' > /etc/apt/sources.list.d/mlnx_ofed.list && \
    wget --no-check-certificate -qO - https://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox | apt-key add -
RUN apt-get update --fix-missing && apt-get install --no-install-recommends -y mlnx-ofed-kernel-utils=23.10.OFED.23.10.2.1.3.1-1 \
    libibverbs-dev=2307mlnx47-1.2310213 \
    && rm -rf /var/lib/apt/lists/

# Install (mini) conda
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

RUN /opt/conda/bin/conda init && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN /opt/conda/bin/conda install -y python="$PYTHON_VERSION"
RUN /opt/conda/bin/conda clean -ya

ENV PATH=/opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"]

ENV POETRY_HOME="/poetry"
RUN mkdir $POETRY_HOME
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=$POETRY_HOME python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN poetry config virtualenvs.path "/poetry/virtualenvs"

# # Install ahm poetry environment
# ENV UNITRAJ_ENV_PATH="/poetry/envs/unitraj"
# RUN mkdir -p $UNITRAJ_ENV_PATH
# COPY poetry/unitraj/ $UNITRAJ_ENV_PATH
# WORKDIR $UNITRAJ_ENV_PATH
# RUN poetry lock --no-interaction
# RUN poetry install --no-interaction --no-root

# Google cloud SDK
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-sdk -y

RUN echo "%users ALL = (ALL) NOPASSWD: ALL" >> /etc/sudoers

USER root