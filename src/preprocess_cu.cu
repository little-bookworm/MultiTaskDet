#include "preprocess_cu.h"

namespace ParkingPerception
{
namespace MultiTaskDet
{
        __global__ void preprocess_kernel(uint8_t *img_device, float *buffer, int batch, int width, int height)
        {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            int j = blockDim.y * blockIdx.y + threadIdx.y;
            int k = blockDim.z * blockIdx.z + threadIdx.z;
            if (i >= width || j >= height || k >= batch)
                return;
            // printf("%d,%d,%d\n", i, j, k);//打印会非常慢
            uint8_t *img = img_device + k * width * height * 3;
            float *tensor = buffer + k * width * height * 3;
            tensor[2 * width * height + width * j + i] = (img[3 * width * j + 3 * i] - 103.53) * 0.01742919; // b
            tensor[width * height + width * j + i] = (img[3 * width * j + 3 * i + 1] - 116.28) * 0.017507;   // g
            tensor[width * j + i] = (img[3 * width * j + 3 * i + 2] - 123.675) * 0.01712475;                 // r
        }

        void preprocess_cu(uint8_t *img_device, float *buffer, int batch, int width, int height, cudaStream_t stream)
        {
            dim3 block_size(32, 32, 1); // blocksize最大就是1024
            dim3 grid_size((width + 31) / 32, (height + 31) / 32, (batch + 0) / 1);

            preprocess_kernel<<<grid_size, block_size, 0, stream>>>(img_device, buffer, batch, width, height);
        }
}  // namespace MultiTaskDet
}  // namespace ParkingPerception