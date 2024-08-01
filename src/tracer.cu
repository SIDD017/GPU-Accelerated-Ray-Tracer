#include "tracer.cuh"
#include "defines.cuh"

namespace CUDA_Tracer {

Tracer::Tracer(int nx, int ny) {
    this->nx = nx;
    this->ny = ny;
    this->num_pixels = nx * ny;
    fb_size = num_pixels * sizeof(vec3);
    CHECK_CUDA_ERRORS(cudaMallocManaged((void **)&fb, fb_size));
}

Tracer::~Tracer() {}

void Tracer::draw(int tx, int ty, cudaGraphicsResource_t resource) {
    uint32_t* dev_ptr;
    size_t pbo_size;
    CHECK_CUDA_ERRORS(cudaGraphicsMapResources(1, &resource));
    // NOTE: resource pointer generated here for the opengl buffer is only accessible by device code (GPU side), 
    // trying to access it on the host will result in crash
    CHECK_CUDA_ERRORS(cudaGraphicsResourceGetMappedPointer((void**)&(dev_ptr), &(pbo_size), resource));
    // std::cerr << "Rendering a " << nx << "x" << ny << " image ";
    // std::cerr << " in " << tx << "x" << ty << " blocks.\n";
    clock_t start, stop;
    start = clock();
    dim3 blocks(nx/tx+1, ny/ty+1);
    dim3 threads(tx, ty);
    render<<<blocks, threads>>>(nx, ny, fb, dev_ptr, 
                                vec3(-2.0, -1.0, -1.0),
                                vec3(4.0, 0.0, 0.0),
                                vec3(0.0, 2.0, 0.0),
                                vec3(0.0, 0.0, 0.0));
    CHECK_CUDA_ERRORS(cudaGetLastError());
    CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    // std::cerr << "took" << timer_seconds << " seconds.\n";
    // output_image();
    CHECK_CUDA_ERRORS(cudaGraphicsUnmapResources(1, &resource));
}

__device__ bool hit_sphere(const vec3& center, float radius, const ray& r) {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0f * dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - 4.0f * a * c;
    return (discriminant > 0.0f);
}

__device__ vec3 color(const ray& r) {
    if(hit_sphere(vec3(0, 0, -1), 0.5f, r)) {
        return vec3(1, 0, 0);
    }
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t*vec3(0.5f, 0.7f, 1.0f);
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
    uint32_t finalb = 0x00000000 | (u32b << 16);
    uint32_t finalg = 0x00000000 | (u32g << 8);
    uint32_t finalr = 0x00000000 | (u32r);
    return 0xFF000000 | finalb | finalg | finalr;
}

__global__
void render(int nx, int ny, vec3 *fb, uint32_t* dev_ptr, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= nx) || (j >= ny)) {
         return;
    }
    int pixel_index = j * nx + i;
    float u = float(i) / float(nx);
    float v = float(j) / float(ny);
    ray r(origin, lower_left_corner + u*horizontal + v*vertical);
    fb[pixel_index] = color(r);
    //Ideally would use these glm functions, but this always returning 0 for some reason
    // dev_ptr[pixel_index] = glm::packUnorm4x8(glm::vec4(fb[pixel_index].r(), fb[pixel_index].g(), fb[pixel_index].b(), 1.0f));
    dev_ptr[pixel_index] = rgb_to_uint_32(fb[pixel_index]);
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