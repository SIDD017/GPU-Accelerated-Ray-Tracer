#ifndef VEC3_H
#define VEC3_H

#include <iostream>
#include <math.h>
#include <stdlib.h>

namespace CUDA_Tracer {
class vec3 {
    public:

    float e[3];

    __host__ __device__ vec3() {}
    __host__ __device__ vec3(float e0, float e1, float e2) {
        e[0] = e0;
        e[1] = e1;
        e[2] = e2;
    }
    __host__ __device__ inline float x() const {return e[0];}
    __host__ __device__ inline float y() const {return e[1];}
    __host__ __device__ inline float z() const {return e[2];}
    __host__ __device__ inline float r() const {return e[0];}
    __host__ __device__ inline float g() const {return e[1];}
    __host__ __device__ inline float b() const {return e[2];}

    __host__ __device__ inline const vec3& operator+() const {return *this;}
    __host__ __device__ inline const vec3& operator-() const {return vec3(-e[0], -e[1], -e[2]);}
    __host__ __device__ inline float operator[](int i) const {return e[i];}
    __host__ __device__ inline float& operator[](int i) {return e[i];}

    __host__ __device__ inline vec3& operator+=(const vec3 &v2);
    __host__ __device__ inline vec3& operator-=(const vec3 &v2);
    __host__ __device__ inline vec3& operator*=(const vec3 &v2);
    __host__ __device__ inline vec3& operator/=(const vec3 &v2);
    __host__ __device__ inline vec3& operator*=(const float t);
    __host__ __device__ inline vec3& operator/=(const float t);
    
    __host__ __device__ inline float length() const {return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);}
    __host__ __device__ inline float squared_length() const {return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];}
    __host__ __device__ inline void make_unit_vector();
};
}

#endif