#include "tracer.cuh"


using namespace CUDA_Tracer;

int main() {
    int nx =  1280;
    int ny = 720;
    int tx = 8;
    int ty = 8;

    CUDA_Tracer::Tracer *tracer = new CUDA_Tracer::Tracer(nx, ny);
    tracer->draw(tx, ty);

    return 0;
}