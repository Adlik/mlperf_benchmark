FROM ubuntu:bionic

RUN apt-get update && \
    apt-get install --no-install-recommends -y ca-certificates wget && \
    wget 'https://apt.repos.intel.com/openvino/2021/GPG-PUB-KEY-INTEL-OPENVINO-2021' -O /etc/apt/trusted.gpg.d/openvino.asc && \
    apt-get clean && \
    find /var/lib/apt/lists -delete

RUN echo 'deb https://apt.repos.intel.com/openvino/2021 all main' >> /etc/apt/sources.list

RUN . /etc/os-release && \
    apt-get update && \
    apt-get install --no-install-recommends -y "intel-openvino-ie-rt-core-$ID-$VERSION_CODENAME-2021.3.394" && \
    apt-get clean && \
    find /var/lib/apt/lists -delete

RUN apt-get update && \
    apt-get install --no-install-recommends -y git llvm-10-dev \
        python3 python3-dev python3-setuptools gcc libtinfo-dev \
        zlib1g-dev build-essential cmake libedit-dev libxml2-dev && \
    apt-get clean && \
    find /var/lib/apt/lists -delete

RUN git clone --recursive https://github.com/apache/tvm tvm && \
    cd tvm && git checkout v0.7 && git submodule update && \
    mkdir build && cp cmake/config.cmake build && \
    cd build && sed -i -E 's/USE_LLVM\s+OFF/USE_LLVM llvm-config-10/g' config.cmake && \
    cmake .. && make -j16
ENV TVM_HOME=/tvm
ENV PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}

RUN apt-get update && \
    apt-get install --no-install-recommends -y python3-pip libgl1-mesa-glx numactl && \
    apt-get clean && \
    find /var/lib/apt/lists -delete

RUN python3 -m pip install -U pip
RUN python3 -m pip install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN python3 -m pip install setuptools wheel openvino decorator absl-py opencv-python
