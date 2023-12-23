#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <iomanip>

using namespace std;

typedef double Type;

const int N = 201;
int M = 0;
const Type T = 20;
const Type tau = T / (N - 1);
const Type G = 6.67*1e-11;
const int block_size = 128;
const Type eps = 0.00001;


struct point {
    Type x;
    Type y;
    Type z;
};

typedef vector<point> vec_point;
typedef vector<Type> vec_Type;

__device__ point operator-(const point& point1, const point& point2) {
    return { point1.x - point2.x, point1.y - point2.y, point1.z - point2.z };
}
__device__ point operator+(const point& point1, const point& point2) {
    return { point1.x + point2.x, point1.y + point2.y, point1.z + point2.z };
}
__device__ point operator*(const point &point, Type a) {
    return { point.x * a, point.y * a, point.z * a };
}
__device__ Type norm2(const point& point) {
    return point.x * point.x + point.y * point.y + point.z * point.z;
}
__device__ Type degree3(const Type x) {
    return x * x * x;
}

void Initialization(const string& name, vec_point& dist, vec_point& speed, vec_Type& mass) {
    ifstream file(name);
    file >> M;
    dist.resize(M);
    speed.resize(M);
    mass.resize(M);

    for (int i = 0; i < M; ++i) {
        file >> mass[i];
        file >> dist[i].x >> dist[i].y >> dist[i].z;
        file >> speed[i].x >> speed[i].y >> speed[i].z;
    }
    file.close();
}

void Clear() {
    for (int i = 0; i < M; ++i) {
        ofstream out;
        out.open("result/traj" + to_string(i + 1) + ".txt");
        out.clear();
        out.close();
    }
}

void Result(const point* dist) {
    for (int i = 0; i < M; ++i) {
        ofstream out;
        out.open("result/traj" + to_string(i + 1) + ".txt", ios::app);
        out << setprecision(15) << dist[i].x << " " << dist[i].y << " " << dist[i].z << "\n";
        out.close();
    }
}

__global__ void calculate(point* dev_dist, point* dev_speed, const point* dev_k2_dist, const point* dev_k2_speed, int M) {
    int my_global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (my_global_id < M) {
        dev_dist[my_global_id] = dev_dist[my_global_id] + dev_k2_dist[my_global_id] * tau;
        dev_speed[my_global_id] = dev_speed[my_global_id] + dev_k2_speed[my_global_id] * tau;
    }
}

__global__ void fun_k1_speed(const Type* dev_mass, const point* dev_dist, point* dev_k1_speed, int M) {

    int my_global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int my_local_id = threadIdx.x;

    __shared__ Type dev_mass_shared[block_size];
    __shared__ point dev_dist_shared[block_size];

    point my_body;
    if (my_global_id < M)
        my_body = dev_dist[my_global_id];

    point result = { 0., 0., 0. };

    for (int i = 0; i < M; i += blockDim.x) {
        dev_mass_shared[my_local_id] = dev_mass[i + my_local_id];
        dev_dist_shared[my_local_id] = dev_dist[i + my_local_id];
        __syncthreads();

        for (int j = 0; j < blockDim.x; ++j)
            if (i + j < M)
                result = result + (my_body - dev_dist_shared[j]) * (1./degree3(sqrt(max(norm2(my_body - dev_dist_shared[j]), degree3(eps)))) * dev_mass_shared[j] * (-G));
        __syncthreads();
    }
    if (my_global_id < M)
        dev_k1_speed[my_global_id] = result;
}

__global__ void fun_k2_dist(const point* dev_speed, const point* dev_k1_speed, point* dev_k2_dist, int M) {
    int my_global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (my_global_id < M)
        dev_k2_dist[my_global_id] = dev_speed[my_global_id] + dev_k1_speed[my_global_id] * 0.5 * tau;
    __syncthreads();
}

__global__ void fun_k2_speed(const Type* dev_mass, const point* dev_dist, point* dev_k2_speed, const point* dev_k1_dist, int M) {
    int my_global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int my_local_id = threadIdx.x;

    __shared__ Type dev_mass_shared[block_size];
    __shared__ point dev_dist_shared[block_size];
    __shared__ point dev_k1_dist_shared[block_size];

    point my_body;
    point k1_my_body;
    if (my_global_id < M) {
        my_body = dev_dist[my_global_id];
        k1_my_body = dev_k1_dist[my_global_id];
    }
    point temp_point;
    point result = { 0., 0., 0. };

    for (int i = 0; i < M; i += blockDim.x) {
        dev_mass_shared[my_local_id] = dev_mass[i + my_local_id];
        dev_dist_shared[my_local_id] = dev_dist[i + my_local_id];
        dev_k1_dist_shared[my_local_id] = dev_k1_dist[i + my_local_id];
        __syncthreads();

        for (int j = 0; j < blockDim.x; ++j) {
            if (i + j < M) {
                if (my_global_id == i + j)
                    temp_point = { 0.,0.,0. };
                else
                    temp_point = (my_body + k1_my_body * 0.5 * tau) - (dev_dist_shared[j] + dev_k1_dist_shared[j] * 0.5 * tau);
                result = result + temp_point * (1./degree3(sqrt(max(norm2(temp_point), degree3(eps))))) * dev_mass_shared[j] * (-G);
            }
        }
        __syncthreads();
    }
    if (my_global_id < M)
        dev_k2_speed[my_global_id] = result;
}

void RungeKutta2(Type* dev_mass, point* dev_dist, point* dev_speed,
    point* dev_k1_dist, point* dev_k2_dist, point* dev_k1_speed, point* dev_k2_speed,
    int block_count, point* res) {

    cudaEvent_t start;
    cudaEvent_t stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for (int i = 1; i < N; ++i) {
        cudaMemcpy(dev_k1_dist, dev_speed, sizeof(point) * M, cudaMemcpyDeviceToDevice);
        fun_k1_speed <<<block_count, block_size >>> (dev_mass, dev_dist, dev_k1_speed, M);
        fun_k2_dist <<<block_count, block_size >>> (dev_speed, dev_k1_speed, dev_k2_dist, M);
        fun_k2_speed <<<block_count, block_size >>> (dev_mass, dev_dist, dev_k2_speed, dev_k1_dist, M);
        calculate <<<block_count, block_size>>>(dev_dist, dev_speed, dev_k2_dist, dev_k2_speed, M);
        cudaDeviceSynchronize();
        cudaMemcpy(res, dev_dist, sizeof(point) * M, cudaMemcpyDeviceToHost);
        Result(res);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    // float
    // printf("N = %d, M = %d, block_size = %d, time = %f", N, M, block_size, time / 1000 / N);
    // double
    printf("N = %d, M = %d, block_size = %d, time = %lf", N, M, block_size, time / 1000 / N);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    vec_point dist;
    vec_point speed;
    vec_Type mass;

    Initialization("init/4body.txt", dist, speed, mass);
    // Initialization("init/init_40000.txt", dist, speed, mass);
    // Initialization("init/init_50000.txt", dist, speed, mass);
    // Initialization("init/init_200000.txt", dist, speed, mass);
    // Initialization("init/init_500000.txt", dist, speed, mass);

    int block_count = M / block_size + static_cast<int>(((M % block_size) != 0));

    cudaSetDevice(0);

    point* dev_dist;
    point* dev_speed;
    Type* dev_mass;
    cudaMalloc(&dev_mass, sizeof(Type) * M);
    cudaMalloc(&dev_dist, sizeof(point) * M);
    cudaMalloc(&dev_speed, sizeof(point) * M);

    point* dev_k1_dist;
    point* dev_k1_speed;
    point* dev_k2_dist;
    point* dev_k2_speed;
    cudaMalloc(&dev_k1_dist, sizeof(point) * M);
    cudaMalloc(&dev_k1_speed, sizeof(point) * M);
    cudaMalloc(&dev_k2_dist, sizeof(point) * M);
    cudaMalloc(&dev_k2_speed, sizeof(point) * M);

    cudaMemcpy(dev_mass, mass.data(), sizeof(Type) * M, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dist, dist.data(), sizeof(point) * M, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_speed, speed.data(), sizeof(point) * M, cudaMemcpyHostToDevice);

    Clear();
    Result(dist.data());
    RungeKutta2(dev_mass, dev_dist, dev_speed, dev_k1_dist, dev_k2_dist, dev_k1_speed, dev_k2_speed, block_count, dist.data());

    cudaFree(dev_mass);
    cudaFree(dev_dist);
    cudaFree(dev_speed);
    cudaFree(dev_k1_dist);
    cudaFree(dev_k1_speed);
    cudaFree(dev_k2_dist);
    cudaFree(dev_k2_speed);
}
