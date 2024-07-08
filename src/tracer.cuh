#ifndef TRACER_H
#define TRACER_H

#include <stdio.h>
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
    void draw(int tx, int ty);
    void output_image();
};

__global__ void render(int nx, int ny, vec3 *fb);
}

#endif