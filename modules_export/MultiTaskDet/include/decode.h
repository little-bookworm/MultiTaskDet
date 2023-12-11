/*
 * @Author: zjj
 * @Date: 2023-12-04 15:12:03
 * @LastEditors: zjj
 * @LastEditTime: 2023-12-05 14:26:16
 * @FilePath: /parking_perception_ros/src/AVM_Perception/architecture/MultiTaskDet/include/decode.h
 * @Description:
 *
 * Copyright (c) 2023 by ${git_name_email}, All Rights Reserved.
 */
#pragma once

#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

namespace ParkingPerception
{
namespace MultiTaskDet
{
void get_det_result(std::vector<std::vector<float>>& box_result, float* tensor, float confidence_threshold,
                    float nms_threshold);

void get_seg_result(cv::Mat& output, int* tensor);
}  // namespace MultiTaskDet
}  // namespace ParkingPerception