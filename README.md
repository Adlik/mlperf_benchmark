# MLPerf Benchmark Tool

[MLPerf](https://mlcommons.org/en/) is a consortium of AI leaders from academia, research labs, and industry whose mission is to “build fair and useful benchmarks” that provide unbiased evaluations of training and inference performance for hardware, software, and services—all conducted under prescribed conditions.

Adlik model optimizer, focusing on and running on specific hardware to achieve the purpose of acceleration, mainly consists of two categories of algorithm components, i.e. pruner and quantizer. The pruner includes various pruning algorithms,  can really reduce the number of parameters and flops. The quantizer focuses on 8-bit quantization that is easier to accelerate on specific hardware.

In this repository MLPerf is used to compare the performance of various pruning and quantization algorithm in Adlik model optimizer.

## 1. Contents

This repository provides the following:

- Code that implements two backends so far, openvino and tvm, more will be provided in the future.
- A Dockerfile which can be used to run the benchmark in a container.
- Documentation on the dataset and  model.
- A test result, based on a specific machine configuration.

## 2. Running Benchmarks

Clone "Adlik/mlperf_benchmark" code repository from github:

```shell
git clone --recurse-submodules https://github.com/Adlik/mlperf_benchmark.git --depth 1
```

Build and install MLPerf loadgen:

```shell
cd third_party/mlperf/loadgen
CFLAGS="-std=c++14 -O3" python3 setup.py bdist_wheel
pip3 install dist/mlperf_loadgen-*-linux_x86_64.whl
```

Prepare the models to be test. first, change directory to the root of this repository,then get the optimized models.

```shell
cd resnet50 && mkdir models && cd models
wget ...
cd ..
```

Prepare the imagenet dataset:

| dataset                   | download link                               |
| ------------------------- | ------------------------------------------- |
| imagenet2012 (validation) | http://image-net.org/challenges/LSVRC/2012/ |

Running the benchmark, following is a example with openvino backend :

```shell
python3 main.py --dataset-path /data/imagenet2012 \
                --model models/resnet50_fp32.zip \
                --profile pytorch-openvino \
                --time 600
```

## 3. Results

The test results of following  table are tested on the following machine configuration:

- Intel(R) Xeon(R) Platinum 8260 CPU @ 2.40GHz.
- Ubuntu 20.04, including docker.
- Greater than 2T of disk (though many benchmarks do require less disk).

| Backend  | Pruning | Quantization | Latency(ms) |
| -------- | ------- | ------------ | ----------- |
| Openvino | ✗       | FP32         | 6.7         |
| Openvino | ✓       | FP32         | 3.3         |
| Openvino | ✗       | INT8         | 2.7         |
| Openvino | ✓       | INT8         | 1.33        |
| TVM      | ✗       | FP32         | 6.7         |
| TVM      | ✓       | FP32         | 2.9         |
| TVM      | ✗       | INT8         | -           |
| TVM      | ✓       | INT8         | -           |
