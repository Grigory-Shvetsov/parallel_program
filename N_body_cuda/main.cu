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
const Type eps3 = eps * eps * eps;

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

__global__ void evaluate(Type* dev_mass, point* dev_dist, point* dev_speed,
    point* dev_k1_dist, point* dev_k2_dist, point* dev_k1_speed, point* dev_k2_speed, int M) {
        

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;
    
    if (global_id < M) {

        __shared__ Type dev_mass_shared[block_size];
        __shared__ point dev_dist_shared[block_size];
        __shared__ point dev_k1_dist_shared[block_size];

        point body;
        body = dev_dist[global_id];

        point result = { 0., 0., 0. };

        for (int i = 0; i < M; i += blockDim.x) {
            dev_mass_shared[local_id] = dev_mass[i + local_id];
            dev_dist_shared[local_id] = dev_dist[i + local_id];
            __syncthreads();
            // __threadfence();

            for (int j = 0; j < blockDim.x; ++j)
                if (i + j < M)
                    result = result + (body - dev_dist_shared[j]) * (1./degree3(sqrt(max(norm2(body - dev_dist_shared[j]), eps3))) * dev_mass_shared[j] * (-G));
            __syncthreads();
            // __threadfence();
        }
        
        dev_k1_speed[global_id] = result;
        dev_k2_dist[global_id] = dev_speed[global_id] + dev_k1_speed[global_id] * 0.5 * tau;
        
        // __syncthreads();
        __threadfence();

        point k1_body;
        k1_body = dev_k1_dist[global_id];
        point temp;
        result = { 0., 0., 0. };

        for (int i = 0; i < M; i += blockDim.x) {
            dev_k1_dist_shared[local_id] = dev_k1_dist[i + local_id];
            __syncthreads();

            for (int j = 0; j < blockDim.x; ++j) {
                if (i + j < M) {
                    if (global_id == i + j)
                        temp = { 0., 0., 0. };
                    else
                        temp = (body + k1_body * 0.5 * tau) - (dev_dist_shared[j] + dev_k1_dist_shared[j] * 0.5 * tau);
                    result = result + temp * (1./degree3(sqrt(max(norm2(temp), eps3)))) * dev_mass_shared[j] * (-G);
                }
            }
            __syncthreads();
        }
        
        dev_k2_speed[global_id] = result;
        
        // __syncthreads();
        // __threadfence();

        dev_dist[global_id] = dev_dist[global_id] + dev_k2_dist[global_id] * tau;
        dev_speed[global_id] = dev_speed[global_id] + dev_k2_speed[global_id] * tau;
        
    }
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
        evaluate<<<block_count, block_size>>>(dev_mass, dev_dist, dev_speed, dev_k1_dist, dev_k2_dist, dev_k1_speed, dev_k2_speed, M);
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

    int block_count = M / block_size + (int)((M % block_size) != 0);

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
