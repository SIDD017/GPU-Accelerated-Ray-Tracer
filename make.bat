cd build
cmake ../
cmake --build .
cd Debug
.\CUDA_Tracer > test.ppm
echo Output ppm generated as : "test.ppm"