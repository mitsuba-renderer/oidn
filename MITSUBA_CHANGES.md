# Changes in the mitsuba-oidn fork

This copy of Intel Open Image Denoise (upstream commit `6602ee2`, version
2.5.1) is vendored by
[mitsuba-oidn](https://github.com/mitsuba-renderer/mitsuba-oidn). It differs
from upstream in one respect: the CPU device runs on an external thread pool
instead of oneTBB, so that it can share the
[nanothread](https://github.com/mitsuba-renderer/nanothread) worker threads of
Dr.Jit and Mitsuba.

The host application installs the pool through a new API function,
`oidnSetThreadPoolAPI()`, which takes a table of nanothread-compatible entry
points. It must be called before creating the first CPU device. The
`devices/cpu/tasking.h` layer maps OIDN's parallel loops onto that pool, and
the thread pinning logic is removed since the pool owns its threads.
Denoising throughput matches the oneTBB build at equal thread counts.

A minor build system addition, the `OIDN_INSTALL_RPATH` CMake variable, sets
the install rpath of all libraries including the CUDA device module, which the
wheel needs to load them from a single directory.
