/*
 * @Author: zjj
 * @Date: 2023-12-04 18:51:43
 * @LastEditors: zjj
 * @LastEditTime: 2023-12-13 14:29:38
 * @FilePath: /MultiTaskDet/include/inference_api.h
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
std::shared_ptr<AVM_MultiTaskDet> CreateDetection(std::string config_file);
}  // namespace MultiTaskDet
}  // namespace ParkingPerception