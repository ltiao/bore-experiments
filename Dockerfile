FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-4
# tensorflow/tensorflow:2.4.2-gpu
MAINTAINER Louis Tiao <louistiao@gmail.com>

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

ARG PIP_VERSION=21.2.3
RUN python -m pip install --no-cache-dir --upgrade pip==${PIP_VERSION}

# Install PyTorch
RUN python -m pip install --no-cache-dir torch==1.9.0+cu111 \
                                         torchvision==0.10.0+cu111 \
                                         torchaudio==0.9.0 \
                                         -f https://download.pytorch.org/whl/torch_stable.html

# Install general dependencies
# TODO: copy to temporary dir rather then some unknown current dir
COPY requirements*.txt ./
COPY src/bore/requirements*.txt /tmp/bore/
RUN python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir -r /tmp/bore/requirements.txt && \
    python -m pip install --no-cache-dir -r /tmp/bore/requirements_dev.txt

# Install BOTorch
ARG GPYTORCH_VERSION=1.5.0
ARG BOTORCH_VERSION=0.5.0
RUN python -m pip install --no-cache-dir gpytorch==${GPYTORCH_VERSION} botorch==${BOTORCH_VERSION}

# Install TPE (hyperopt) and SMAC
ARG HYPEROPT_VERSION=0.2.4
ARG SMAC_VERSION=0.13.1
RUN python -m pip install --no-cache-dir hyperopt==${HYPEROPT_VERSION} smac==${SMAC_VERSION}

# Install HpBandSter and HPO Bench
RUN python -m pip install --no-cache-dir Pyro4 h5py
COPY src/HpBandSter /tmp/HpBandSter
COPY src/nas_benchmarks /tmp/nas_benchmarks
RUN python -m pip install --no-cache-dir --no-deps -e /tmp/HpBandSter && \
    python -m pip install --no-cache-dir --no-deps -e /tmp/nas_benchmarks

# Install GPy and GPyOpt
ARG GPY_VERSION=1.10.0
RUN python -m pip install --no-cache-dir GPy==${GPY_VERSION}
COPY src/GPyOpt /tmp/GPyOpt
RUN python -m pip install --no-cache-dir --no-deps -e /tmp/GPyOpt

# Install BORE
COPY src/bore /tmp/bore
RUN python -m pip install --no-cache-dir --no-deps -e /tmp/bore

RUN mkdir -p /usr/src/app

COPY . /usr/src/app
WORKDIR /usr/src/app
RUN python -m pip install --no-cache-dir --no-deps -e .

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"]
