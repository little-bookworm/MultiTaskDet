/*
 * @Author: zjj
 * @Date: 2023-12-04 18:51:43
 * @LastEditors: zjj
 * @LastEditTime: 2023-12-05 14:26:33
 * @FilePath: /parking_perception_ros/src/AVM_Perception/architecture/MultiTaskDet/include/inference_api.h
 * @Description:
 *
 * Copyright (c) 2023 by ${git_name_email}, All Rights Reserved.
 */
#pragma once

#include "inference.h"

namespace ParkingPerception
{
namespace MultiTaskDet
{
AVM_MultiTaskDet* CreateDetection(std::string config_file);
}  // namespace MultiTaskDet
}  // namespace ParkingPerception