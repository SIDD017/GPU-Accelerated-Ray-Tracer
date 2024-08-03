#include <curand_kernel.h>
#include "tracer.cuh"
#include "defines.cuh"
#include "sphere.cuh"
#include "hitable_list.cuh"
#include "camera.cuh"
#include "material.cuh"
namespace CUDA_Tracer {

Tracer::Tracer(int nx, int ny, int ns) {
    this->nx = nx;
    this->ny = ny;
    this->ns = ns;
    this->num_pixels = nx * ny;
    fb_size = num_pixels * sizeof(vec3);
    // CHECK_CUDA_ERRORS(cudaMallocManaged((void **)&fb, fb_size));
}

Tracer::~Tracer() {}

__device__ bool hit_sphere(const vec3& center, float radius, const ray& r) {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0f * dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - 4.0f * a * c;
    return (discriminant > 0.0f);
}

// #define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

// __device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
//     vec3 p;
//     do {
//         p = 2.0f*RANDVEC3 - vec3(1,1,1);
//     } while (p.squared_length() >= 1.0f);
//     return p;
// }

__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}
 
__device__ uint32_t rgb_to_uint_32(const vec3& col) {
    int b = 255 * col.b();
    int g = 255 * col.g();
    int r = 255 * col.r();
    uint8_t ub = 0x00 + b;
    uint8_t ug = 0x00 + g;
    uint8_t ur = 0x00 + r;
    uint32_t u32b = ub;
    uint32_t u32g = ug;
    uint32_t u32r = ur;
    uint32_t finalb = 0x00000000 | (u32b);
    uint32_t finalg = 0x00000000 | (u32g << 8);
    uint32_t finalr = 0x00000000 | (u32r << 16);
    return 0xFF000000 | finalb | finalg | finalr;
}

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new sphere(vec3(0,0,-1), 0.5,
                               new lambertian(vec3(0.8, 0.3, 0.3)));
        d_list[1] = new sphere(vec3(0,-100.5,-1), 100,
                               new lambertian(vec3(0.8, 0.8, 0.0)));
        d_list[2] = new sphere(vec3(1,0,-1), 0.5,
                               new metal(vec3(0.8, 0.6, 0.2), 1.0));
        d_list[3] = new sphere(vec3(-1,0,-1), 0.5,
                               new metal(vec3(0.8, 0.8, 0.8), 0.3));
        *d_world  = new hitable_list(d_list,4);
        *d_camera = new camera();
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__
void render(int max_x, int max_y, int ns, vec3 *fb, uint32_t* dev_ptr, camera **cam, hitable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u,v);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    dev_ptr[pixel_index] = rgb_to_uint_32(col);
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera) {
    for(int i=0; i < 4; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}


void Tracer::draw(int tx, int ty, cudaGraphicsResource_t resource) {
    uint32_t* dev_ptr;
    size_t pbo_size;
    CHECK_CUDA_ERRORS(cudaGraphicsMapResources(1, &resource));
    // NOTE: resource pointer generated here for the opengl buffer is only accessible by device code (GPU side), 
    // trying to access it on the host will result in crash
    CHECK_CUDA_ERRORS(cudaGraphicsResourceGetMappedPointer((void**)&(dev_ptr), &(pbo_size), resource));
    // std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    // std::cerr << " in " << tx << "x" << ty << " blocks.\n";
    curandState *d_rand_state;
    CHECK_CUDA_ERRORS(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    hitable **d_list;
    CHECK_CUDA_ERRORS(cudaMalloc((void **)&d_list, 4*sizeof(hitable *)));
    hitable **d_world;
    CHECK_CUDA_ERRORS(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    camera **d_camera;
    CHECK_CUDA_ERRORS(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world<<<1,1>>>(d_list,d_world,d_camera);
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    dim3 blocks(nx/tx+1, ny/ty+1);
    dim3 threads(tx, ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(nx, ny, ns, fb, dev_ptr, 
                                d_camera, d_world, d_rand_state);
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    // std::cerr << "took" << timer_seconds << " seconds.\n";
    // output_image();
    CHECK_CUDA_ERRORS(cudaGraphicsUnmapResources(1, &resource));

    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world,d_camera);
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaFree(d_camera));
    CHECK_CUDA_ERRORS(cudaFree(d_world));
    CHECK_CUDA_ERRORS(cudaFree(d_list));
    CHECK_CUDA_ERRORS(cudaFree(d_rand_state));
    // CHECK_CUDA_ERRORS(cudaFree(fb));
}

void Tracer::output_image() {
    std::cout  << "P3\n" << nx << " " << ny << "\n255\n";
    for(int j = ny - 1; j >= 0;j--) {
        for(int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            int ir = int(255.99 * fb[pixel_index].r());
            int ig = int(255.99 * fb[pixel_index].g());
            int ib = int(255.99 * fb[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
}
}