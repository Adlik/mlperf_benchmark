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

- Code that implements four benckmark test cases so far, Resnet-50, BERT, Yolo-v5, and MaskRcnn,
  other models will be provided in the future.
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

### For ResNet50

Prepare the models to be tested. first, change directory to the root of this repository,then get the optimized models.

```shell
cd cnns && mkdir models && cd models
wget ...
cd ..
```

Prepare the imagenet dataset:

| dataset                   | download link                                 |
| ------------------------- | --------------------------------------------- |
| imagenet2012 (validation) | <http://image-net.org/challenges/LSVRC/2012/> |

Running the benchmark, following is a example with OpenVINO backend :

```shell
python3 main.py --dataset-path /data/imagenet2012 \
                --model models/resnet50_fp32.zip \
                --profile pytorch-openvino \
                --time 600
```

### For Bert

Same as resnet50, prepare the model to be tested.

```shell
wget model_to_be_tested -O path_to_model
```

Then prepare the SQuAD dataset:

| dataset    | download link                                                      |
| ---------- | ------------------------------------------------------------------ |
| SQuAD v1.1 | <https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json> |

Finally run the benchmark test by command below:

```shell
source conf/setup_envs_offline.sh
python3 run_xxx.py --batch-size=$BATCH_SIZE \
                          --num-instance=$NUM_INSTANCE \
                          --num-phy-cpus=$NUM_PHY_CPUS \
                          --log-dir=$LOG_DIR \
                          --batching=NaiveBucket \
                          --mlperf-conf=conf/mlperf.conf \
                          --user-conf=conf/user.conf \
                          --path_to_model=path_to_model \
                          --dataset=path_to_data \
                          --scenario=Offline
```

### For Yolo-v5 and MaskRcnn

Same as resnet-50, prepare the model to be tested. We used yolov5m and mask_rcnn_R_50_FPN_1x in results 3.2.

```shell
wget model_to_be_tested -O path_to_model
```

Then prepare the coco dataset:

| dataset      | download link                       |
| ------------ | ----------------------------------- |
| coco val2017 | <https://cocodataset.org/#download> |

```shell
python3 main.py --dataset-path /data/coco \
                --model models/$model \
                --profile coco-$model_type-openvino \
                --time 600 \
                --scenario SingleStream \
                (--accuracy)
```

## 3. Results

### 3.1 Part 1

The inference benchmark tests, including ResNet50 and BERT different backends, is running in a
docker instance on Ubuntu 20.04, while the device information is as follows:

- Intel(R) Xeon(R) Platinum 8260 CPU @ 2.40GHz (2 Sockets).
- Greater than 2T of disk (though many benchmarks do require less disk).

Intel(R) Xeon(R) Platinum 8260 is a 64-bit 24-core x86 high-performance server microprocessor
introduced by Intel in 2019. It's based on Cascadelake microarchitecture and is manufactured on a
14 nm process. The chip supports 8-way multiprocessing, sports 2 AVX-512 FMA units as well as three
Ultra Path Interconnect links.

In practice, our benchmark is running on a ZXCloud R5300 G4 Server,
which is ZTE’s new generation 2U 2-socket rack server, integrating 2 Intel(R) Xeon(R) Platinum 8260.
R5300 G4 uses high-density and modular design, providing high performance, high reliability,
high scalability, easy management and other features, widely applicable to Internet,
cloud computing, big data, NFV, SDN and other fields.

The benchmark results are summarized in the table below. All the testcases are running on 48 physical cores.

#### Resnet50

| Backend  | Pruning | Precision | Latency (ms) |
| -------- | ------- | --------- | ------------ |
| OpenVINO | ✗       | FP32      | 6.7          |
| OpenVINO | ✓       | FP32      | 3.3          |
| TVM      | ✗       | FP32      | 6.7          |
| TVM      | ✓       | FP32      | 2.9          |

#### Bert (Detailed log in bert/mlperf_log/)

| Backend     | Precision | Samples Per Second |
| ----------- | --------- | ------------------ |
| TVM         | FP32      | 16.02              |
| OpenVINO    | FP32      | 11.43              |
| OnnxRuntime | FP32      | 11.28              |

### 3.2 Part 2

The inference benchmark runs with openvino backend, including ResNet-50, Yolo-v5 and MaskRcnn
which are running in a docker instance, while the device information is as follows:

- Intel(R) Xeon(R) Platinum 8378C CPU @ 2.80GHz (2 Sockets)
- Greater than 2T of disk

Intel(R) Xeon(R) Platinum 8378C is a 64-bit 38-core x86 high-performance server microprocessor
introduced by Intel in 2021. It's based on Icelake microarchitecture and is manufactured on a
10 nm process.

#### Resnet-50

| Backend  | Pruning | Type | Latency (ms) | Acc(%) |
| -------- | ------- | ---- | ------------ | ------ |
| OpenVINO | ✗       | FP32 | 4.3          | 76.1   |
| OpenVINO | ✗       | INT8 | 2.0          | 75.9   |
| OpenVINO | ✓       | INT8 | 1.7          | 75.7   |

#### Yolo-v5

| Backend  | Pruning | Type | Latency (ms) | mAP@.5 / mAP@.5:.95 |
| -------- | ------- | ---- | ------------ | ------------------- |
| OpenVINO | ✗       | FP32 | 18.7         | 63.9 / 44.8         |
| OpenVINO | ✗       | INT8 | 7.5          | 63.7 / 44.4         |
| OpenVINO | ✓       | INT8 | 7.1          | 59.6 / 39.9         |

#### MaskRcnn

| Backend  | Pruning | Type | Latency (ms) | mAP@.5 / mAP@.5:.95 (box) | mAP@.5 / mAP@.5:.95 (mask) |
| -------- | ------- | ---- | ------------ | ------------------------- | -------------------------- |
| OpenVINO | ✗       | FP32 | 133.1        | 55.0 / 34.6               | 51.7 / 31.2                |
| OpenVINO | ✗       | INT8 | 89.5         | 55.0 / 34.6               | 51.7 / 31.2                |
