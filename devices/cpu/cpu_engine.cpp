// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "core/conv.h"
#include "core/concat_conv.h"
#include "cpu_engine.h"
#if !defined(OIDN_BNNS)
  #include "cpu_conv.h"
  #if defined(OIDN_ARCH_X64) && !defined(__APPLE__)
    #include "cpu_conv_amx.h"
  #endif
#endif
#include "cpu_pool.h"
#include "cpu_upsample.h"
#include "cpu_autoexposure.h"
#include "cpu_input_process.h"
#include "cpu_output_process.h"
#include "cpu_image_copy.h"

#if defined(__APPLE__)
  #include <pthread/qos.h>
#endif

OIDN_NAMESPACE_BEGIN

  CPUEngine::CPUEngine(CPUDevice* device, int numThreads)
    : device(device)
  {
    // Use the shared thread pool unless a specific thread count was requested
    arena = std::make_shared<TaskArena>(numThreads);

    // Start the queue processing thread
    queueThread = std::thread([&]() { processQueue(); });
  }

  CPUEngine::~CPUEngine()
  {
    {
      std::lock_guard<std::mutex> lock(queueMutex);
      queueShutdown = true;
    }
    queueCond.notify_all();
    queueThread.join();
  }

  bool CPUEngine::isConvSupported(Fusion fusion) const
  {
    if (device->getArch() == CPUArch::AVX512_AMXFP16)
    {
      return fusion == Fusion::None ||
             fusion == Fusion::UpsampleSrc0 ||
             fusion == Fusion::PoolDst;
    }
    else
      return fusion == Fusion::None;
  }

  bool CPUEngine::isConcatConvSupported(Fusion fusion) const
  {
    if (device->getArch() == CPUArch::AVX512_AMXFP16)
    {
      return fusion == Fusion::None ||
             fusion == Fusion::UpsampleSrc0 ||
             fusion == Fusion::PoolDst;
    }
    else
      return fusion == Fusion::None;
  }

#if !defined(OIDN_BNNS)
  Ref<Conv> CPUEngine::newConv(const ConvDesc& desc)
  {
  #if defined(OIDN_ARCH_X64) && !defined(__APPLE__)
    if (device->getArch() == CPUArch::AVX512_AMXFP16)
      return makeRef<CPUConvAMX>(this, desc);
    else
  #endif
      return makeRef<CPUConv>(this, desc);
  }
#endif

  Ref<ConcatConv> CPUEngine::newConcatConv(const ConcatConvDesc& desc)
  {
  #if defined(OIDN_ARCH_X64) && !defined(__APPLE__)
    if (device->getArch() == CPUArch::AVX512_AMXFP16)
      return makeRef<CPUConcatConvAMX>(this, desc);
    else
  #endif
    return makeRef<PreConcatConvCHW>(this, desc);
  }

  Ref<Pool> CPUEngine::newPool(const PoolDesc& desc)
  {
    return makeRef<CPUPool>(this, desc);
  }

  Ref<Upsample> CPUEngine::newUpsample(const UpsampleDesc& desc)
  {
    return makeRef<CPUUpsample>(this, desc);
  }

  Ref<Autoexposure> CPUEngine::newAutoexposure(const ImageDesc& srcDesc)
  {
    return makeRef<CPUAutoexposure>(this, srcDesc);
  }

  Ref<InputProcess> CPUEngine::newInputProcess(const InputProcessDesc& desc)
  {
    return makeRef<CPUInputProcess>(this, desc);
  }

  Ref<OutputProcess> CPUEngine::newOutputProcess(const OutputProcessDesc& desc)
  {
    return makeRef<CPUOutputProcess>(this, desc);
  }

  Ref<ImageCopy> CPUEngine::newImageCopy()
  {
    return makeRef<CPUImageCopy>(this);
  }

  void CPUEngine::submitFunc(std::function<void()>&& f, const Ref<CancellationToken>& ct)
  {
    {
      std::lock_guard<std::mutex> lock(queueMutex);
      queue.push({std::move(f), ct});
    }
    queueCond.notify_all();
  }

  void CPUEngine::submitHostFunc(std::function<void()>&& f, const Ref<CancellationToken>& ct)
  {
    submitFunc(std::move(f), ct);
  }

  void CPUEngine::wait()
  {
    {
      std::unique_lock<std::mutex> lock(queueMutex);
      queueCond.wait(lock, [&] { return queue.empty(); });
    }
  }

  void* CPUEngine::usmAlloc(size_t byteSize, Storage storage)
  {
    if (storage != Storage::Host && storage != Storage::Device && storage != Storage::Managed)
      throw Exception(Error::InvalidArgument, "invalid storage mode");

    if (byteSize == 0)
      return nullptr;
    return alignedMalloc(byteSize);
  }

  void CPUEngine::usmFree(void* ptr, Storage storage)
  {
    if (ptr != nullptr)
      alignedFree(ptr);
  }

  void CPUEngine::usmCopy(void* dstPtr, const void* srcPtr, size_t byteSize)
  {
    submitUSMCopy(dstPtr, srcPtr, byteSize);
    wait();
  }

  void CPUEngine::submitUSMCopy(void* dstPtr, const void* srcPtr, size_t byteSize)
  {
    submitFunc([=] { std::memcpy(dstPtr, srcPtr, byteSize); });
  }

  void CPUEngine::processQueue()
  {
  #if defined(__APPLE__)
    // Match the QoS class of the thread pool workers, which this thread drives and joins
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
  #endif

    for (; ;)
    {
      Task task;

      // Wait until a task is available in the queue
      {
        std::unique_lock<std::mutex> lock(queueMutex);
        queueCond.wait(lock, [&] { return !queue.empty() || queueShutdown; });
        if (queue.empty() && queueShutdown)
          return;
        task = std::move(queue.front());
      }

      // Execute queued tasks in the arena until the queue gets empty
      arena->execute([&]
      {
        for (; ;)
        {
          if (task.ct && task.ct->isCancelled())
            device->setAsyncError(Error::Cancelled, "execution was cancelled");
          else
            task.func();

          task = {};

          {
            std::lock_guard<std::mutex> lock(queueMutex);
            queue.pop();
            if (queue.empty())
              break;
            task = std::move(queue.front());
          }
        }

        queueCond.notify_all();
      });
    }
  }

OIDN_NAMESPACE_END
