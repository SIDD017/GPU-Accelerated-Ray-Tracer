# GPU accelerated Real-Time Ray tracer

This project follows from the CPU based implemenation of a simple [path tracer](https://github.com/SIDD017/Ray_Tracer) created using C++. While it served as a great introduction to ray tracing and graphics programming in general, the base implementation was extremely inefficient as core computations were sequential, resulting in extremely long render times (nearly 1 hour to render a single frame).

This project aims to implement a more efficient real-time ray tracer using CUDA for GPU acceleration and OpenGL for rendering on screen. To maintain portability, CMake is utilized as the build system.
