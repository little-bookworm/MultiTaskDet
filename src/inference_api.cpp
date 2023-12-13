/*
 * @Author: zjj
 * @Date: 2023-12-04 18:52:07
 * @LastEditors: zjj
 * @LastEditTime: 2023-12-13 14:30:54
 * @FilePath: /MultiTaskDet/src/inference_api.cpp
 * @Description:
 *
 * Copyright (c) 2023 by ${git_name_email}, All Rights Reserved.
 */
#include "inference_api.h"

namespace ParkingPerception
{
namespace MultiTaskDet
{
std::shared_ptr<AVM_MultiTaskDet> CreateDetection(std::string config_file)
{
  return std::make_shared<AVM_MultiTaskDet>(config_file);
}
}  // namespace MultiTaskDet
}  // namespace ParkingPerception