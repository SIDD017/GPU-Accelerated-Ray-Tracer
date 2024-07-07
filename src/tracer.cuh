#ifndef TRACER_H
#define TRACER_H

#include <stdio.h>

namespace CUDA_Tracer {
class Tracer {
    private:
    int nx;
    int ny;
    int num_pixels;
    size_t fb_size;
    float *fb;

    public:
    Tracer(int nx, int ny);
    ~Tracer();
    void draw(int tx, int ty);
    void output_image();
};

__global__ void render(int nx, int ny, float *fb);
}

#endif