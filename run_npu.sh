cd build/mooncake-transfer-engine/example
./transfer_engine_heterogeneous_ascend_perf_initiator \
    --mode=initiator \
    --local_server_name=7.6.16.155 \
    --metadata_server=P2PHANDSHAKE \
    --operation=write \
    --npu_id=0 \
    --segment_id=7.6.16.150:16901 \
    --device_name=ibp56s0 \
    --block_size=1024 \
    --batch_size=5242880