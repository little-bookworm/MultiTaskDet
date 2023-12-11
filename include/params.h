/*
 * @Author: zjj
 * @Date: 2023-12-04 11:08:02
 * @LastEditors: zjj
 * @LastEditTime: 2023-12-05 14:26:52
 * @FilePath: /parking_perception_ros/src/AVM_Perception/architecture/MultiTaskDet/include/params.h
 * @Description:
 *
 * Copyright (c) 2023 by ${git_name_email}, All Rights Reserved.
 */
#pragma once

#include <vector>
#include <string>

namespace ParkingPerception
{
namespace MultiTaskDet
{
struct Params
{
  Params()
    : batch_size(1)
    , img_h(640)
    , img_w(640)
    , model_file("")
    , confidence_threshold(0.25)
    , nms_threshold(0.45)
    , tensorNum(4)
    , tensorNames(std::vector<std::string>{})
    , tensorDim0(std::vector<int>{})
    , tensorDim1(std::vector<int>{})
    , tensorDim2(std::vector<int>{})
    , tensorDim3(std::vector<int>{})
    , device_id(0){};
  int batch_size;
  int img_h;
  int img_w;
  std::string model_file;
  float confidence_threshold;
  float nms_threshold;
  int tensorNum;
  std::vector<std::string> tensorNames;
  std::vector<int> tensorDim0;
  std::vector<int> tensorDim1;
  std::vector<int> tensorDim2;
  std::vector<int> tensorDim3;
  int device_id;
  Params operator=(const Params& other)
  {
    this->batch_size = other.batch_size;
    this->img_h = other.img_h;
    this->img_w = other.img_w;
    this->model_file = other.model_file;
    this->confidence_threshold = other.confidence_threshold;
    this->nms_threshold = other.nms_threshold;
    this->tensorNum = other.tensorNum;
    this->tensorNames = other.tensorNames;
    this->tensorDim0 = other.tensorDim0;
    this->tensorDim1 = other.tensorDim1;
    this->tensorDim2 = other.tensorDim2;
    this->tensorDim3 = other.tensorDim3;
    this->device_id = other.device_id;
    return *this;
  }
};
}  // namespace MultiTaskDet
}  // namespace ParkingPerception
