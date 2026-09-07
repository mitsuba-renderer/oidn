// Copyright 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_conv_amx.h"
#include "cpu_common.h"
#include "cpu_conv_amx_ispc.h"

OIDN_NAMESPACE_BEGIN

  oidn_inline void submitCPUConvAMXKernel(CPUEngine* engine, const ispc::CPUConvAMXKernel& kernel,
                                          const Ref<CancellationToken>& ct)
  {
    // Block sizes of the kernel
    constexpr int blockOC = 32; // output channel block size
    constexpr int blockOW = 16; // output block width before post-op
    constexpr int blockOH = 2;  // output block height before post-op

    constexpr int chunkOH = 8; // we group output height blocks into larger chunks

    engine->submitFunc([=]
    {
      const int OC = kernel.dst.C; // output channels
      const int OH = kernel.OH;    // output height before post-op
      const int OW = kernel.OW;    // output width before post-op

      const int OCB = OC / blockOC;          // number of output channel blocks
      const int OHC = ceil_div(OH, chunkOH); // number of output height chunks
      const int OWB = ceil_div(OW, blockOW); // number of output width blocks

      const size_t N = size_t(OCB) * OHC * OWB; // total number of work items

      parallel_for_range(N, [&](size_t begin, size_t end)
      {
        for (size_t i = begin; i != end; ++i)
        {
          const size_t j = i / OCB;
          const int ocb = int(i % OCB); // output channel block index
          const int owb = int(j % OWB); // output width block index
          const int ohc = int(j / OWB); // output height chunk index

          const int oc = ocb * blockOC;
          const int ow = owb * blockOW;
          const int ohBegin = round_up(ohc * OH / OHC, blockOH);
          const int ohEnd   = min(round_up((ohc + 1) * OH / OHC, blockOH), OH);

          const bool isFirst = (i == begin);
          const bool isLast  = (i == end - 1);

          ispc::CPUConvAMXKernel_run_f16(&kernel, oc, ohBegin, ohEnd, ow, isFirst, isLast);
        }
      });
    }, ct);
  }

  CPUConvAMX::CPUConvAMX(CPUEngine* engine, const ConvDesc& desc)
    : Conv(desc),
      engine(engine)
  {
    if (srcDesc.layout != TensorLayout::Chw32c || srcDesc.dataType != DataType::Float16)
      throw std::invalid_argument("unsupported convolution source layout/data type");
    if (weightDesc.getW() != 3 || weightDesc.getH() != 3)
      throw std::invalid_argument("unsupported convolution kernel size");
    if (weightDesc.layout != TensorLayout::OIhw2o16i16o2i || weightDesc.dataType != DataType::Float16)
      throw std::invalid_argument("unsupported convolution weight layout/data type");
    if (biasDesc.layout != TensorLayout::x || biasDesc.dataType != DataType::Float16)
      throw std::invalid_argument("unsupported convolution bias layout/data type");
  }

  void CPUConvAMX::submitKernels(const Ref<CancellationToken>& ct)
  {
    if (!src || !dst || !weight || !bias)
      throw std::logic_error("convolution argument not set");

    ispc::CPUConvAMXKernel kernel;
    ispc::CPUConvAMXKernel_init(&kernel);

    kernel.src[0]  = *src;
    kernel.src[1]  = {}; // unused
    kernel.weight  = *weight;
    kernel.bias    = *bias;
    kernel.dst     = *dst;
    kernel.numSrcs = 1;
    kernel.relu    = activation == Activation::ReLU;

    switch (fusion)
    {
    case Fusion::None:
      kernel.fusion = ispc::Fusion_None;
      kernel.OH = dst->getH();
      kernel.OW = dst->getW();
      break;

    case Fusion::UpsampleSrc0:
      kernel.fusion = ispc::Fusion_UpsampleSrc0;
      kernel.OH = dst->getH();
      kernel.OW = dst->getW();
      break;

    case Fusion::PoolDst:
      kernel.fusion = ispc::Fusion_PoolDst;
      kernel.OH = src->getH();
      kernel.OW = src->getW();
      break;

    default:
      throw std::logic_error("unsupported convolution fusion");
    }

    submitCPUConvAMXKernel(engine, kernel, ct);
  }

  CPUConcatConvAMX::CPUConcatConvAMX(CPUEngine* engine, const ConcatConvDesc& desc)
    : ConcatConv(desc),
      engine(engine)
  {
    if (src0Desc.layout != TensorLayout::Chw32c || src0Desc.dataType != DataType::Float16)
      throw std::invalid_argument("unsupported convolution source layout/data type");
    if (src1Desc.layout != TensorLayout::Chw32c || src1Desc.dataType != DataType::Float16)
      throw std::invalid_argument("unsupported convolution source layout/data type");
    if (weightDesc.getW() != 3 || weightDesc.getH() != 3)
      throw std::invalid_argument("unsupported convolution kernel size");
    if (weightDesc.layout != TensorLayout::OIhw2o16i16o2i || weightDesc.dataType != DataType::Float16)
      throw std::invalid_argument("unsupported convolution weight layout/data type");
    if (biasDesc.layout != TensorLayout::x || biasDesc.dataType != DataType::Float16)
      throw std::invalid_argument("unsupported convolution bias layout/data type");
  }

  void CPUConcatConvAMX::submitKernels(const Ref<CancellationToken>& ct)
  {
    if (!src0 || !src1 || !dst || !weight || !bias)
      throw std::logic_error("concat+convolution argument not set");

    ispc::CPUConvAMXKernel kernel;
    ispc::CPUConvAMXKernel_init(&kernel);

    kernel.src[0]  = *src0;
    kernel.src[1]  = *src1;
    kernel.weight  = *weight;
    kernel.bias    = *bias;
    kernel.dst     = *dst;
    kernel.numSrcs = 2;
    kernel.OH      = src1->getH();
    kernel.OW      = src1->getW();
    kernel.relu    = activation == Activation::ReLU;

    switch (fusion)
    {
    case Fusion::None:          kernel.fusion = ispc::Fusion_None;         break;
    case Fusion::UpsampleSrc0:  kernel.fusion = ispc::Fusion_UpsampleSrc0; break;
    case Fusion::PoolDst:       kernel.fusion = ispc::Fusion_PoolDst;      break;
    default:
      throw std::logic_error("unsupported concat+convolution fusion");
    }

    submitCPUConvAMXKernel(engine, kernel, ct);
  }

OIDN_NAMESPACE_END