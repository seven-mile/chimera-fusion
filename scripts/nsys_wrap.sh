#!/bin/bash

if [[ -z "${NSYS_OUTPUT}" ]]; then
    NSYS_OUTPUT=prof
fi
if [[ -z "${NSYS_NODE_INTERVAL}" ]]; then
    NSYS_NODE_INTERVAL=1
fi

NSYS_BIN=/opt/app/cuda/12.2/bin/nsys

# TODO: capture all ranks
if [ "${SLURM_LOCALID}" -eq 0 ] && [ "$(( SLURM_NODEID % NSYS_NODE_INTERVAL ))" -eq 0 ];
then
    $NSYS_BIN profile \
        -f true \
        -o ${NSYS_OUTPUT}_node${SLURM_NODEID} \
        -c cudaProfilerApi \
        --trace cuda,nvtx,cudnn,osrt \
        --export sqlite \
        $@ 2>&1 | tee output/logs/rank${SLURM_PROCID}.log
else
    $@ 2>&1 | tee output/logs/rank${SLURM_PROCID}.log
fi
sleep 30  # wait for nsys to complete
