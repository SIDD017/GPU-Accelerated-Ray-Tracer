#ifndef SPHERE_H
#define SPHERE_H

#include "hitable.cuh"

namespace CUDA_Tracer {

class sphere : public hitable {
    public:
    float radius;
    vec3 center;
    __device__ sphere(){}
    __device__ sphere(vec3 cen, float r): center(cen), radius(r) {}
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;
    if(discriminant > 0) {
        float temp = (-b - sqrt(discriminant)) / a;
        if(temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_At_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if(temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_At_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            return true;
        }
    }
    return false;
}

}

#endif