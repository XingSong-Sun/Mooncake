// Copyright 2025 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "transport/cuda_heterogeneous_transport/cuda_heterogeneous_rdma_transport.h"

#include <chrono>

namespace mooncake {

namespace {
bool isCpuMemory(void *addr) {
    cudaPointerAttributes attributes{};
    cudaError_t ret = cudaPointerGetAttributes(&attributes, addr);
    if (ret != cudaSuccess) {
        // If CUDA cannot identify the pointer, it is not CUDA-managed device
        // memory, so treat it as regular CPU memory.
        LOG(WARNING) << "cudaPointerGetAttributes failed for addr " << addr
                     << ", ret: " << cudaGetErrorString(ret)
                     << ". Treating as CPU memory.";
        cudaGetLastError();  // Clear the error state
        return true;
    }
    // cudaMemoryTypeDevice indicates the memory resides on the GPU device.
    // All other types (Host, Unregistered, Managed) are accessible from CPU.
    return (attributes.type != cudaMemoryTypeDevice);
}
}  // namespace

CudaHeterogeneousRdmaTransport::~CudaHeterogeneousRdmaTransport() {
    running_ = false;
    transfer_queue_cv_.notify_one();
    transfer_thread_.join();
    if (host_addr_) {
        cudaFreeHost(host_addr_);
        host_addr_ = nullptr;
    }
    if (dev_addr_) {
        cudaFree(dev_addr_);
        dev_addr_ = nullptr;
    }
    if (stream_copy_created_) {
        cudaStreamDestroy(stream_copy_);
        stream_copy_created_ = false;
    }
}

void CudaHeterogeneousRdmaTransport::transferLoop() {
    Status s;
    cudaStream_t stream_d2h{};
    cudaError_t ret = cudaSetDevice(device_id_);
    if (ret != cudaSuccess) {
        LOG(ERROR) << "CudaHeterogeneousRdmaTransport: cudaSetDevice error, "
                      "ret: "
                   << cudaGetErrorString(ret);
    }

    ret = cudaStreamCreate(&stream_d2h);
    if (ret != cudaSuccess) {
        LOG(ERROR)
            << "CudaHeterogeneousRdmaTransport: cudaStreamCreate error, ret: "
            << cudaGetErrorString(ret);
    }

    while (running_) {
        auto transfer_info = getTransfer();
        const auto &task_list = transfer_info.tasks;
        if (task_list.empty()) {
            if (!running_) break;
            LOG(ERROR)
                << "CudaHeterogeneousRdmaTransport: empty transfer task batch";
            continue;
        }

        int total_length = transfer_info.total_length;
        if ((host_offset_ + total_length) > CUDA_HETERO_HOST_SIZE) {
            host_offset_ = 0;
        }

        ret = cudaMemcpyAsync(host_addr_ + host_offset_, transfer_info.block,
                              total_length, cudaMemcpyDeviceToHost, stream_d2h);
        if (ret != cudaSuccess) {
            LOG(ERROR) << "CudaHeterogeneousRdmaTransport: cudaMemcpyAsync "
                          "dtoh error, ret: "
                       << cudaGetErrorString(ret)
                       << ", hostAddr: " << (void *)host_addr_
                       << ", host_offset_: " << host_offset_
                       << ", deviceAddr: " << (void *)transfer_info.block
                       << ", len: " << total_length;
            continue;
        }

        ret = cudaStreamSynchronize(stream_d2h);
        if (ret != cudaSuccess) {
            LOG(ERROR)
                << "CudaHeterogeneousRdmaTransport: cudaStreamSynchronize "
                   "error, ret: "
                << cudaGetErrorString(ret);
            continue;
        }

        for (size_t index = 0; index < task_list.size(); ++index) {
            auto &task = *task_list[index];
            auto &request = *task.request;
            request.source = host_addr_ + host_offset_;
            host_offset_ += request.length;
        }

        s = transport_->submitTransferTask(task_list);
        if (!s.ok()) {
            LOG(ERROR) << "CudaHeterogeneousRdmaTransport: Rdma "
                          "submitTransferTask error";
        }
        releaseBlock(transfer_info.block);
    }

    cudaStreamDestroy(stream_d2h);
}

int CudaHeterogeneousRdmaTransport::install(
    std::string &local_server_name, std::shared_ptr<TransferMetadata> meta,
    std::shared_ptr<Topology> topo) {
    local_server_name_ = local_server_name;
    metadata_ = meta;  // Must set metadata_ on this instance before use
    running_ = true;

    cudaError_t ret = cudaGetDevice(&device_id_);
    if (ret != cudaSuccess) {
        LOG(ERROR)
            << "CudaHeterogeneousRdmaTransport: cudaGetDevice failed, ret: "
            << cudaGetErrorString(ret);
        return -1;
    }

    LOG(INFO) << "CudaHeterogeneousRdmaTransport: begin to install transport, "
                 "local_server_name: "
              << local_server_name << ", device_id_: " << device_id_;

    if (transport_ == nullptr) {
        LOG(ERROR) << "CudaHeterogeneousRdmaTransport: transport is null";
        return -1;
    }

    // Allocate pinned host memory for staging buffer
    ret = cudaMallocHost((void **)&host_addr_, CUDA_HETERO_HOST_SIZE);
    if (ret != cudaSuccess) {
        LOG(ERROR)
            << "CudaHeterogeneousRdmaTransport: cudaMallocHost failed, ret: "
            << cudaGetErrorString(ret);
        return -1;
    }

    // Allocate device memory for aggregation blocks
    dev_addr_ = nullptr;
    ret = cudaMalloc((void **)&dev_addr_,
                     (size_t)CUDA_HETERO_DEVICE_NUM * CUDA_HETERO_DEVICE_SIZE);
    if (ret != cudaSuccess) {
        LOG(ERROR)
            << "CudaHeterogeneousRdmaTransport: cudaMalloc failed, ret: "
            << cudaGetErrorString(ret);
        cudaFreeHost(host_addr_);
        host_addr_ = nullptr;
        return -1;
    }

    for (int i = 0; i < CUDA_HETERO_DEVICE_NUM; i++) {
        block_queue_.push(dev_addr_ + i * CUDA_HETERO_DEVICE_SIZE);
    }

    transfer_thread_ =
        std::thread(&CudaHeterogeneousRdmaTransport::transferLoop, this);

    int install_ret = transport_->install(local_server_name_, meta, topo);
    if (install_ret) {
        LOG(ERROR) << "CudaHeterogeneousRdmaTransport: RdmaTransport install "
                      "error, ret: "
                   << install_ret;
        return install_ret;
    }

    install_ret = transport_->registerLocalMemory(
        host_addr_, CUDA_HETERO_HOST_SIZE, "cpu", true, true);
    if (install_ret) {
        LOG(ERROR) << "CudaHeterogeneousRdmaTransport: registerLocalMemory "
                      "error, ret: "
                   << install_ret;
        return install_ret;
    }
    return 0;
}

int CudaHeterogeneousRdmaTransport::registerLocalMemory(
    void *addr, size_t length, const std::string &name, bool remote_accessible,
    bool update_metadata) {
    if (isCpuMemory(addr)) {
        if (int ret = transport_->registerLocalMemory(addr, length, "cpu", true,
                                                      true)) {
            LOG(ERROR) << "rdma transport registerLocalMemory error, ret: "
                       << ret;
            return ret;
        }
    }
    return 0;
}

int CudaHeterogeneousRdmaTransport::unregisterLocalMemory(
    void *addr, bool update_metadata) {
    if (isCpuMemory(addr)) {
        int ret = transport_->unregisterLocalMemory(addr, true);
        if (ret) {
            LOG(ERROR) << "rdma transport unregisterLocalMemory error, ret: "
                       << ret;
            return ret;
        }
    }
    return 0;
}

int CudaHeterogeneousRdmaTransport::registerLocalMemoryBatch(
    const std::vector<CudaHeterogeneousRdmaTransport::BufferEntry>
        &buffer_list,
    const std::string &location) {
    for (auto &buffer : buffer_list) {
        int ret = registerLocalMemory(buffer.addr, buffer.length, location,
                                      true, false);
        if (ret) {
            LOG(ERROR) << "CudaHeterogeneousRdmaTransport "
                          "registerLocalMemoryBatch error, ret: "
                       << ret;
            return ret;
        }
    }
    return metadata_->updateLocalSegmentDesc();
}

int CudaHeterogeneousRdmaTransport::unregisterLocalMemoryBatch(
    const std::vector<void *> &addr_list) {
    for (auto &addr : addr_list) {
        int ret = unregisterLocalMemory(addr, false);
        if (ret) {
            LOG(ERROR) << "CudaHeterogeneousRdmaTransport "
                          "unregisterLocalMemoryBatch error, ret: "
                       << ret;
            return ret;
        }
    }
    return metadata_->updateLocalSegmentDesc();
}

int CudaHeterogeneousRdmaTransport::checkAndCreateStreamCopy() {
    if (!stream_copy_created_) {
        cudaError_t ret = cudaSetDevice(device_id_);
        if (ret != cudaSuccess) {
            LOG(ERROR)
                << "CudaHeterogeneousRdmaTransport: cudaSetDevice failed, "
                   "ret: "
                << cudaGetErrorString(ret);
            return -1;
        }
        ret = cudaStreamCreate(&stream_copy_);
        if (ret != cudaSuccess) {
            LOG(ERROR)
                << "CudaHeterogeneousRdmaTransport: cudaStreamCreate error, "
                   "ret: "
                << cudaGetErrorString(ret);
            return -1;
        }
        stream_copy_created_ = true;
    }
    return 0;
}

Status CudaHeterogeneousRdmaTransport::submitTransfer(
    BatchID batch_id, const std::vector<TransferRequest> &entries) {
    if (entries.empty()) {
        LOG(ERROR)
            << "CudaHeterogeneousRdmaTransport: empty transfer request batch";
        return Status::OK();
    }

    if (isCpuMemory(entries[0].source)) {
        return transport_->submitTransfer(batch_id, entries);
    }

    std::vector<TransferRequest> new_entries;
    new_entries.resize(entries.size());
    int index = 0;
    int ret = checkAndCreateStreamCopy();
    if (ret) {
        LOG(ERROR)
            << "CudaHeterogeneousRdmaTransport: createStream error, ret: "
            << ret;
        return Status::InvalidArgument(
            "CudaHeterogeneousRdmaTransport: createStream error");
    }

    {
        std::lock_guard<std::mutex> lock(memcpy_mutex_);
        for (auto &request : entries) {
            if (host_offset_ + request.length > CUDA_HETERO_HOST_SIZE) {
                host_offset_ = 0;
            }
            cudaError_t cuda_ret = cudaMemcpyAsync(
                host_addr_ + host_offset_, request.source, request.length,
                cudaMemcpyDeviceToHost, stream_copy_);
            if (cuda_ret != cudaSuccess) {
                LOG(ERROR)
                    << "CudaHeterogeneousRdmaTransport: cudaMemcpyAsync "
                       "error, ret: "
                    << cudaGetErrorString(cuda_ret)
                    << ", hostAddr: " << (void *)host_addr_
                    << ", host_offset_: " << host_offset_
                    << ", deviceAddr: " << request.source
                    << ", len: " << request.length;
                return Status::InvalidArgument(
                    "CudaHeterogeneousRdmaTransport: cudaMemcpyAsync error");
            }
            new_entries[index] = request;
            new_entries[index].source = host_addr_ + host_offset_;
            host_offset_ += request.length;
        }

        cudaError_t cuda_ret = cudaStreamSynchronize(stream_copy_);
        if (cuda_ret != cudaSuccess) {
            LOG(ERROR)
                << "CudaHeterogeneousRdmaTransport: cudaStreamSynchronize "
                   "error, ret: "
                << cudaGetErrorString(cuda_ret);
            return Status::InvalidArgument(
                "CudaHeterogeneousRdmaTransport: cudaStreamSynchronize error");
        }
    }

    return transport_->submitTransfer(batch_id, new_entries);
}

Status CudaHeterogeneousRdmaTransport::noAggTransport(
    const std::vector<TransferTask *> &task_list) {
    for (size_t index = 0; index < task_list.size(); ++index) {
        auto &task = *task_list[index];
        auto &request = *task.request;

        if (host_offset_ + request.length > CUDA_HETERO_HOST_SIZE) {
            host_offset_ = 0;
        }

        cudaError_t ret =
            cudaMemcpyAsync(host_addr_ + host_offset_, request.source,
                            request.length, cudaMemcpyDeviceToHost, stream_copy_);
        if (ret != cudaSuccess) {
            LOG(ERROR)
                << "CudaHeterogeneousRdmaTransport: cudaMemcpyAsync error, "
                   "ret: "
                << cudaGetErrorString(ret)
                << ", hostAddr: " << (void *)host_addr_
                << ", host_offset_: " << host_offset_
                << ", deviceAddr: " << request.source
                << ", len: " << request.length;
            return Status::InvalidArgument(
                "CudaHeterogeneousRdmaTransport: cudaMemcpyAsync error");
        }
        request.source = host_addr_ + host_offset_;
        host_offset_ += request.length;
    }

    cudaError_t ret = cudaStreamSynchronize(stream_copy_);
    if (ret != cudaSuccess) {
        LOG(ERROR)
            << "CudaHeterogeneousRdmaTransport: cudaStreamSynchronize error, "
               "ret: "
            << cudaGetErrorString(ret);
        return Status::InvalidArgument(
            "CudaHeterogeneousRdmaTransport: cudaStreamSynchronize error");
    }

    return transport_->submitTransferTask(task_list);
}

Status CudaHeterogeneousRdmaTransport::aggTransport(
    const std::vector<TransferTask *> &task_list) {
    uint64_t index = 0;
    while (index < task_list.size()) {
        auto block = acquireBlock();
        uint64_t block_offset = 0;
        std::vector<TransferTask *> tasks;

        while (index < task_list.size()) {
            auto &request = *(task_list[index]->request);
            if (block_offset + request.length > CUDA_HETERO_DEVICE_SIZE) {
                break;
            }

            cudaError_t ret = cudaMemcpyAsync(
                block + block_offset, request.source, request.length,
                cudaMemcpyDeviceToDevice, stream_copy_);
            if (ret != cudaSuccess) {
                LOG(ERROR) << "CudaHeterogeneousRdmaTransport: "
                              "cudaMemcpyAsync dtod error, ret: "
                           << cudaGetErrorString(ret)
                           << ", host_offset_: " << host_offset_
                           << ", deviceAddr: " << request.source
                           << ", len: " << request.length;
                return Status::InvalidArgument(
                    "CudaHeterogeneousRdmaTransport: cudaMemcpyAsync dtod "
                    "error");
            }

            tasks.push_back(task_list[index]);
            block_offset += request.length;
            ++index;
        }

        cudaError_t ret = cudaStreamSynchronize(stream_copy_);
        if (ret != cudaSuccess) {
            LOG(ERROR) << "CudaHeterogeneousRdmaTransport: "
                          "cudaStreamSynchronize failed, ret: "
                       << cudaGetErrorString(ret);
            return Status::InvalidArgument(
                "CudaHeterogeneousRdmaTransport: cudaStreamSynchronize error");
        }

        addTransfer(std::move(tasks), block, block_offset);
    }

    auto start = std::chrono::high_resolution_clock::now();
    while (!allBlockReleased()) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        auto count = duration_ms.count();
        if (count > 100) {
            LOG(ERROR) << "CudaHeterogeneousRdmaTransport: transfer_queue_ "
                          "wait too long, count:"
                       << count << "ms";
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    return Status::OK();
}

Status CudaHeterogeneousRdmaTransport::submitTransferTask(
    const std::vector<TransferTask *> &task_list) {
    if (task_list.empty()) {
        LOG(ERROR)
            << "CudaHeterogeneousRdmaTransport: empty transfer task list";
        return Status::InvalidArgument(
            "CudaHeterogeneousRdmaTransport: task_list is empty");
    }

    auto &task = *task_list[0];
    auto &request = *task.request;

    if (isCpuMemory(request.source)) {
        return transport_->submitTransferTask(task_list);
    }

    auto ret = checkAndCreateStreamCopy();
    if (ret) {
        LOG(ERROR)
            << "CudaHeterogeneousRdmaTransport: createStream error, ret: "
            << ret;
        return Status::InvalidArgument(
            "CudaHeterogeneousRdmaTransport: createStream error");
    }

    {
        std::lock_guard<std::mutex> lock(memcpy_mutex_);
        if (request.length >= CUDA_HETERO_AGGREGATE_SIZE_LIMIT) {
            return noAggTransport(task_list);
        } else {
            return aggTransport(task_list);
        }
    }

    return Status::OK();
}

Status CudaHeterogeneousRdmaTransport::getTransferStatus(
    BatchID batch_id, std::vector<TransferStatus> &status) {
    return transport_->getTransferStatus(batch_id, status);
}

Status CudaHeterogeneousRdmaTransport::getTransferStatus(
    BatchID batch_id, size_t task_id, TransferStatus &status) {
    return transport_->getTransferStatus(batch_id, task_id, status);
}

}  // namespace mooncake
