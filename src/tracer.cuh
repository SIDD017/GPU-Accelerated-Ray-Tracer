#ifndef TRACER_H
#define TRACER_H

#include <stdio.h>
//Remove from app.cuh maybe
#include <glm/glm.hpp>
#include "vec3.cuh"

namespace CUDA_Tracer {
class Tracer {
    private:
    int nx;
    int ny;
    int num_pixels;
    size_t fb_size;
    vec3 *fb;

    public:
    Tracer(int nx, int ny);
    ~Tracer();
    void draw(int tx, int ty, cudaGraphicsResource_t resource);
    void output_image();
};

__global__ void render(int nx, int ny, vec3 *fb, uint32_t* dev_ptr);
__global__ void pbo_test(int nx, int ny, uint32_t* dev);
}

#endif