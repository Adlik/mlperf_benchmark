FROM ubuntu:bionic
SHELL ["/bin/bash","-c"]

RUN apt-get update && \
    apt-get install -y g++ wget git

RUN cd /tmp && \
    apt-get install -y gnupg2 && \
    wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB && \
    apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB && \
    rm GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB && \
    echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list && \
    apt-get update && \
    apt-get install -y intel-mkl-2019.5-075

RUN apt-get install -y python3.7-dev python3-pip && \
    python3.7 -m pip install pip && \
    python3.7 -m pip install cmake absl-py

# Install LoadGen
RUN cd /tmp && \
    git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference && \
    cd mlperf_inference && \
    git checkout r1.0  && \
    git log -1 && \
    git submodule update --init --recursive && \
    cd loadgen && \
    CFLAGS="-std=c++14" python3.7 setup.py install

RUN source /opt/intel/compilers_and_libraries/linux/mkl/bin/mklvars.sh intel64

# install gluonnlp
RUN cd /tmp && \
    git clone https://github.com/dmlc/gluon-nlp.git && \
    cd gluon-nlp && \
    git checkout 0d5c61992180f41eab590e74c7b679980f429292 && \
    git log -1 && \
    python3.7 setup.py develop

# Install dependencies
RUN python3 -m pip install torch==1.4.0 onnx==1.6.0 transformers==2.4.0 onnxruntime==1.2.0 numpy==1.18.0
RUN apt-get install -y pkg-config libhdf5-dev && \
    python3.7 -m pip install tensorflow==1.14.0rc1 transformers

ENV LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:$LD_LIBRARY_PATH

RUN cd /tmp && \
    git clone https://github.com/mlcommons/inference_results_v1.0.git && \
    cd inference_results_v1.0/closed/Intel/calibration/MXNet/bert/ && \
    sed -i "s#mxnet_path = pathlib.Path.*#mxnet_path = pathlib.Path(\"/tmp/incubator-mxnet\")#" bertpass_setup.py && \
    python3.7 bertpass_setup.py install

# Add user
RUN apt-get install -y protobuf-compiler libprotobuf-dev && \
    python3.7 -m pip install onnx onnxruntime tokenization