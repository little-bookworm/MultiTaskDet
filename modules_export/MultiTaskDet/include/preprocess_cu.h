/*
 * @Author: zjj
 * @Date: 2023-12-04 14:53:54
 * @LastEditors: zjj
 * @LastEditTime: 2023-12-05 14:26:58
 * @FilePath: /parking_perception_ros/src/AVM_Perception/architecture/MultiTaskDet/include/preprocess_cu.h
 * @Description:
 *
 * Copyright (c) 2023 by ${git_name_email}, All Rights Reserved.
 */
#pragma once

// #include <iostream>
#include "stdio.h"
#include <cuda_runtime.h>

typedef unsigned char uint8_t;

namespace ParkingPerception
{
namespace MultiTaskDet
{
__global__ void preprocess_kernel(uint8_t* img_device, float* buffer, int batch, int width, int height);

void preprocess_cu(uint8_t* img_device, float* buffer, int batch, int width, int height, cudaStream_t stream);
}  // namespace MultiTaskDet
}  // namespace ParkingPerception