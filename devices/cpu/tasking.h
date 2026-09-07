// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/context.h"
#include <algorithm>
#include <vector>

OIDN_NAMESPACE_BEGIN

  // -----------------------------------------------------------------------------------------------
  // Thread pool access
  // -----------------------------------------------------------------------------------------------

  // Returns the installed thread pool API, throwing if none was installed
  const OIDNThreadPoolAPI& getThreadPoolAPI();

  // Thread pool used by parallel loops on the calling thread (nullptr = shared default pool)
  void* getCurrentThreadPool();
  void setCurrentThreadPool(void* pool);

  // -----------------------------------------------------------------------------------------------
  // TaskArena
  // -----------------------------------------------------------------------------------------------

  // Thread pool that executes the parallel loops of a CPU engine. With numThreads <= 0, this is the
  // shared default pool of the thread pool library. Otherwise, a private pool is created.
  class TaskArena
  {
  public:
    explicit TaskArena(int numThreads);
    ~TaskArena();

    // Number of threads available for parallel work, including the calling thread
    int max_concurrency() const;

    // Runs f with this pool selected for parallel loops on the calling thread
    template<typename F>
    void execute(F&& f)
    {
      void* prevPool = getCurrentThreadPool();
      setCurrentThreadPool(pool);
      try
      {
        f();
      }
      catch (...)
      {
        setCurrentThreadPool(prevPool);
        throw;
      }
      setCurrentThreadPool(prevPool);
    }

  private:
    TaskArena(const TaskArena&) = delete;
    TaskArena& operator =(const TaskArena&) = delete;

    void* pool = nullptr;
    bool owned = false;
  };

  // -----------------------------------------------------------------------------------------------
  // Parallel loops
  // -----------------------------------------------------------------------------------------------

  // Number of blocks to split a loop of the given size into. Threads claim blocks dynamically,
  // so a generous count per thread balances the load at negligible cost.
  template<typename T>
  oidn_inline T getNumParallelBlocks(T size)
  {
    const T numThreads = T(getThreadPoolAPI().poolSize(getCurrentThreadPool()));
    return std::max(T(1), std::min(size, numThreads * 16));
  }

  // Splits [0, size) into contiguous blocks and invokes f(blockIndex, begin, end) for each of them
  template<typename T, typename F>
  void parallel_for_blocks(const T& size, const F& f)
  {
    if (size <= T(0))
      return;

    const T numBlocks = getNumParallelBlocks(size);
    if (numBlocks == T(1))
    {
      f(T(0), T(0), size);
      return;
    }

    struct Payload
    {
      const F* f;
      T size;
      T numBlocks;
    };

    Payload payload{&f, size, numBlocks};

    auto callback = [](uint32_t i, void* p)
    {
      const Payload* pl = static_cast<const Payload*>(p);
      const T begin = T(i) * pl->size / pl->numBlocks;
      const T end   = T(i + 1) * pl->size / pl->numBlocks;
      (*pl->f)(T(i), begin, end);
    };

    const OIDNThreadPoolAPI& api = getThreadPoolAPI();
    void* task = api.taskSubmitDep(getCurrentThreadPool(), nullptr, 0, uint32_t(numBlocks),
                                   callback, &payload, 0, nullptr, 0, 0);
    api.taskWaitAndReleaseExclusive(task);
  }

  // Splits [0, size) into contiguous blocks and invokes f(begin, end) for each of them
  template<typename T, typename F>
  oidn_inline void parallel_for_range(const T& size, const F& f)
  {
    parallel_for_blocks(size, [&](T, T begin, T end) { f(begin, end); });
  }

  // Invokes f(i) for i in [0, D0)
  template<typename T0, typename F>
  oidn_inline void parallel_for(const T0& D0, const F& f)
  {
    parallel_for_range(D0, [&](T0 begin, T0 end)
    {
      for (T0 i = begin; i != end; ++i)
        f(i);
    });
  }

  // Invokes f(i, j) for i in [0, D0) and j in [0, D1)
  template<typename T0, typename T1, typename F>
  oidn_inline void parallel_for(const T0& D0, const T1& D1, const F& f)
  {
    using T = decltype(D0 * D1);
    const T size = T(D0) * T(D1);

    parallel_for_range(size, [&](T begin, T end)
    {
      for (T k = begin; k != end; ++k)
        f(T0(k / T(D1)), T1(k % T(D1)));
    });
  }

  // Deterministic parallel reduction over [0, size). Each block accumulates into its own copy of
  // identity via f(i, value), and the block results are combined in block order.
  template<typename T, typename V, typename F, typename C>
  V parallel_deterministic_reduce(const T& size, const V& identity, const F& f, const C& combine)
  {
    if (size <= T(0))
      return identity;

    std::vector<V> partials(size_t(getNumParallelBlocks(size)), identity);

    parallel_for_blocks(size, [&](T block, T begin, T end)
    {
      V value = identity;
      for (T i = begin; i != end; ++i)
        f(i, value);
      partials[size_t(block)] = value;
    });

    V result = identity;
    for (const V& value : partials)
      result = combine(result, value);
    return result;
  }

OIDN_NAMESPACE_END
