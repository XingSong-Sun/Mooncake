cd build/mooncake-transfer-engine/tests
./rdma_transport_test \
    --mode=target \
    --local_server_name=7.6.16.150 \
    --metadata_server=P2PHANDSHAKE \
    --operation=write \
    --protocol=rdma \
    --device_name=ibp22s0 \
    --use_vram=true \
    --gpu_id=2