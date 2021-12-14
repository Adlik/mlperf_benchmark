# MLPerf Benchmark Tool

[MLPerf](https://mlcommons.org/en/) is a consortium of AI leaders from academia, research labs, and industry whose
mission is to “build fair and useful benchmarks” that provide unbiased evaluations of training and inference
performance for hardware, software, and services—all conducted under prescribed conditions.

Adlik model optimizer, focusing on and running on specific hardware to achieve the purpose of acceleration, mainly
consists of two categories of algorithm components, i.e. pruner and quantizer. The pruner includes various pruning
 algorithms,  can really reduce the number of parameters and flops. The quantizer focuses on 8-bit quantization
 that is easier to accelerate on specific hardware.

In this repository MLPerf is used to compare the performance of various pruning and quantization algorithm in Adlik
model optimizer.

## 1. Contents

This repository provides the following:

- Code that implements two benckmark test cases so far, Resnet-50 and BERT, more will be provided in the future.
- Dockerfiles which can be used to run the benchmark in a container.
- Documentation on the dataset and model.
- Test results, based on a specific machine configuration, including resnet50 and BERT testcases.

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

### For resnet50

Prepare the models to be tested. first, change directory to the root of this repository,then get the optimized models.

```shell
cd resnet50 && mkdir models && cd models
wget ...
cd ..
```

Prepare the imagenet dataset:

| dataset                   | download link                               |
| ------------------------- | ------------------------------------------- |
| imagenet2012 (validation) | <http://image-net.org/challenges/LSVRC/2012/> |

Running the benchmark, following is a example with OpenVINO backend :

```shell
python3 main.py --dataset-path /data/imagenet2012 \
                --model models/resnet50_fp32.zip \
                --profile pytorch-openvino \
                --time 600
```

### For BERT

Same as resnet50, prepare the model to be tested.

```shell
wget model_to_be_tested -O path_to_model
```

Then prepare the SQuAD dataset:

| dataset                   | download link                               |
| ------------------------- | ------------------------------------------- |
| SQuAD v1.1 | <https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json> |

Finally run the benchmark test by command below:

```shell
source conf/setup_vars_offline.sh
python3 run_bert_bench.py --batch-size=$BATCH_SIZE \
                          --num-instance=$NUM_INSTANCE \
                          --num-phy-cpus=$NUM_PHY_CPUS \
                          --log-dir=$LOG_DIR \
                          --batching=NaiveBucket \
                          --mlperf-conf=conf/mlperf.conf \
                          --user-conf=conf/user.conf \
                          --path_to_model=path_to_model \
                          --scenario=Offline
```

## 3. Results

The following table is the results of the inference benchmark tests, including ResNet-50 and BERT different backends,
while the machine configuration is as follows:

- Intel(R) Xeon(R) Platinum 8260 CPU @ 2.40GHz (2 Sockets).
- Ubuntu 20.04, including docker.
- Greater than 2T of disk (though many benchmarks do require less disk).

### Resnet-50

| Backend  | Pruning | Precision | Latency (ms) |
| -------- | ------- | ------------ | ----------- |
| OpenVINO | ✗       | FP32         | 6.7         |
| OpenVINO | ✓       | FP32         | 3.3         |
| TVM      | ✗       | FP32         | 6.7         |
| TVM      | ✓       | FP32         | 2.9         |

### BERT

| Backend  |  Precision    | Samples Per Second |
| -------- |  ------------ | ----------- |
| TVM      | FP32          | 16.02       |
| OpenVINO | FP32          | 11.43       |
