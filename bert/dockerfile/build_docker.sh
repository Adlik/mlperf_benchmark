#!/bin/bash
docker build --build-arg http_proxy=http://localhost:8080 \
	--build-arg https_proxy=http://localhost:8080 \
    --network=host \
	-t mlperf-bert-benchmark .
