#ifndef RAY_H
#define RAY_H
#include "vec3.cuh"


namespace CUDA_Tracer {

class ray {
    public:
    __device__ ray() {}
    __device__ ray(const vec3& a, const vec3& b) {A = a; B = b;}
    __device__ vec3 origin() const {return A;}
    __device__ vec3 direction() const {return B;}
    __device__ vec3 point_At_parameter(float t) {return A + t*B;}

    private:
    vec3 A;
    vec3 B;
};
}

#endif