## Copyright 2018 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

find_package(Threads REQUIRED)

if(APPLE AND OIDN_ARCH STREQUAL "ARM64")
  # BNNS
  option(OIDN_DEVICE_CPU_BNNS "Enable BNNS for CPU device." ON)
  mark_as_advanced(OIDN_DEVICE_CPU_BNNS)
  if(OIDN_DEVICE_CPU_BNNS)
    set(OIDN_BNNS ON)
    include(oidn_bnns)
  endif()
endif()

# ISPC
if(OIDN_ARCH STREQUAL "X64")
  if(APPLE)
    set(OIDN_ISPC_TARGET_LIST sse4.1-i32x8 avx2-i32x8 avx512skx-x16)
  else()
    set(OIDN_ISPC_TARGET_LIST sse4.1-i32x8 avx2-i32x8 avx512skx-x16 avx512gnr-x16)
  endif()
elseif(OIDN_ARCH STREQUAL "ARM64")
  set(OIDN_ISPC_TARGET_LIST neon-i32x8)
endif()
set(ISPC_ADDRESSING 64)
include(oidn_ispc)

set(OIDN_CPU_SOURCES
  cpu_autoexposure.h
  cpu_autoexposure.cpp
  cpu_common.h
  cpu_common.cpp
  cpu_device.h
  cpu_device.cpp
  cpu_engine.h
  cpu_engine.cpp
  cpu_image_copy.h
  cpu_image_copy.cpp
  cpu_input_process.h
  cpu_input_process.cpp
  cpu_module.cpp
  cpu_output_process.h
  cpu_output_process.cpp
  cpu_pool.h
  cpu_pool.cpp
  cpu_upsample.h
  cpu_upsample.cpp
  tasking.h
  tasking.cpp
)

set(OIDN_ISPC_SOURCES_COMMON
  cpu_autoexposure.ispc
  cpu_image_copy.ispc
  cpu_input_process.isph
  cpu_input_process_f32.ispc
  cpu_output_process.isph
  cpu_output_process_f32.ispc
  cpu_pool.isph
  cpu_pool_f32.ispc
  cpu_upsample.isph
  cpu_upsample_f32.ispc
  color.isph
  color.ispc
  image_accessor.isph
  math.isph
  platform.isph
  tensor_accessor.isph
  tile.isph
  vec.isph
)

if(OIDN_BNNS)
  list(APPEND OIDN_CPU_SOURCES
    bnns/bnns_common.h
    bnns/bnns_common.cpp
    bnns/bnns_conv.h
    bnns/bnns_conv.cpp
    bnns/bnns_engine.h
    bnns/bnns_engine.cpp
    bnns/bnns_pool.h
    bnns/bnns_pool.cpp
  )
else()
  list(APPEND OIDN_CPU_SOURCES
    cpu_conv.h
    cpu_conv.cpp
  )

  if(OIDN_ARCH STREQUAL "X64" AND NOT APPLE)
    list(APPEND OIDN_CPU_SOURCES
      cpu_conv_amx.h
      cpu_conv_amx.cpp
    )
  endif()

  list(APPEND OIDN_ISPC_SOURCES_COMMON
    cpu_conv.ispc
  )
endif()

add_library(OpenImageDenoise_device_cpu ${OIDN_LIB_TYPE} ${OIDN_CPU_SOURCES} ${OIDN_RESOURCE_FILE})

set_target_properties(OpenImageDenoise_device_cpu PROPERTIES
  OUTPUT_NAME ${OIDN_LIBRARY_NAME}_device_cpu
)
if(OIDN_LIBRARY_VERSIONED)
  set_target_properties(OpenImageDenoise_device_cpu PROPERTIES VERSION ${PROJECT_VERSION})
endif()

if(OIDN_BNNS)
  target_compile_definitions(OpenImageDenoise_device_cpu PRIVATE OIDN_BNNS)
  ispc_add_definitions(-DOIDN_BNNS)
  target_link_libraries(OpenImageDenoise_device_cpu PRIVATE "-framework Accelerate")
endif()

if(OIDN_ARCH STREQUAL "X64")
  ispc_target_add_sources(OpenImageDenoise_device_cpu
    platform.ispc
    TARGET_LIST sse2-i32x4 ${OIDN_ISPC_TARGET_LIST} # SSE2 added just for ISA detection
  )
else()
  list(APPEND OIDN_ISPC_SOURCES_COMMON platform.ispc)
endif()

ispc_target_add_sources(OpenImageDenoise_device_cpu
  ${OIDN_ISPC_SOURCES_COMMON}
  TARGET_LIST ${OIDN_ISPC_TARGET_LIST}
)

if(OIDN_ARCH STREQUAL "X64" AND NOT APPLE)
  ispc_target_add_sources(OpenImageDenoise_device_cpu
    cpu_input_process_f16.ispc
    cpu_output_process_f16.ispc
    TARGET_LIST avx512gnr-x16
  )

  ispc_target_add_sources(OpenImageDenoise_device_cpu
    cpu_conv_amx.ispc
    TARGET_LIST avx512gnr-x16
    COMPILE_OPTIONS --enable-llvm-intrinsics # AMX requires LLVM intrinsics
  )

  ispc_target_add_sources(OpenImageDenoise_device_cpu
    cpu_pool_f16.ispc
    cpu_upsample_f16.ispc
    TARGET_LIST avx512gnr-x32
  )
elseif(OIDN_ARCH STREQUAL "ARM64")
  ispc_target_add_sources(OpenImageDenoise_device_cpu
    cpu_input_process_f16.ispc
    cpu_output_process_f16.ispc
    TARGET_LIST neon-i32x8
  )

  ispc_target_add_sources(OpenImageDenoise_device_cpu
    cpu_pool_f16.ispc
    cpu_upsample_f16.ispc
    TARGET_LIST neon-i16x16
  )
else()
  ispc_target_add_sources(OpenImageDenoise_device_cpu
    cpu_input_process_f16.ispc
    cpu_output_process_f16.ispc
    TARGET_LIST avx512skx-x16
  )

  ispc_target_add_sources(OpenImageDenoise_device_cpu
    cpu_pool_f16.ispc
    cpu_upsample_f16.ispc
    TARGET_LIST avx512skx-x32
  )
endif()

target_link_libraries(OpenImageDenoise_device_cpu PRIVATE OpenImageDenoise_core ${CMAKE_THREAD_LIBS_INIT})

if(OIDN_STATIC_LIB)
  oidn_install_static_module(OpenImageDenoise_device_cpu)
  target_link_libraries(OpenImageDenoise PRIVATE OpenImageDenoise_device_cpu)
else()
  oidn_strip_symbols(OpenImageDenoise_device_cpu)
  oidn_install_module(OpenImageDenoise_device_cpu)
endif()
