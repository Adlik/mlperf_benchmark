export BATCH_SIZE=64
export NUM_INSTANCE=6
export CPUS_PER_INSTANCE=8
export OMP_NUM_THREADS=$CPUS_PER_INSTANCE
export DNNL_PRIMITIVE_CACHE_CAPACITY=10485760
export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact
export NUM_PHY_CPUS=$(( $NUM_INSTANCE*$CPUS_PER_INSTANCE ))
