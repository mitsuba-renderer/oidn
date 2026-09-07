// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tasking.h"

OIDN_NAMESPACE_BEGIN

  static thread_local void* currentThreadPool = nullptr;

  const OIDNThreadPoolAPI& getThreadPoolAPI()
  {
    const OIDNThreadPoolAPI* api = Context::get().getThreadPoolAPI();
    if (api == nullptr)
      throw Exception(Error::InvalidOperation, "no thread pool installed (see oidnSetThreadPoolAPI)");
    return *api;
  }

  void* getCurrentThreadPool()
  {
    return currentThreadPool;
  }

  void setCurrentThreadPool(void* pool)
  {
    currentThreadPool = pool;
  }

  TaskArena::TaskArena(int numThreads)
  {
    const OIDNThreadPoolAPI& api = getThreadPoolAPI();
    if (numThreads > 0)
    {
      pool = api.poolCreate(uint32_t(numThreads), 1);
      if (pool == nullptr)
        throw Exception(Error::Unknown, "failed to create thread pool");
      owned = true;
    }
  }

  TaskArena::~TaskArena()
  {
    if (owned)
      getThreadPoolAPI().poolDestroy(pool);
  }

  int TaskArena::max_concurrency() const
  {
    return int(getThreadPoolAPI().poolSize(pool));
  }

OIDN_NAMESPACE_END
