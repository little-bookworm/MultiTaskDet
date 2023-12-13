/*
 * @Author: zjj
 * @Date: 2023-12-04 10:55:58
 * @LastEditors: zjj
 * @LastEditTime: 2023-12-13 13:37:38
 * @FilePath: /MultiTaskDet/include/inference.h
 * @Description:
 *
 * Copyright (c) 2023 by ${git_name_email}, All Rights Reserved.
 */
#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <yaml-cpp/yaml.h>

#include "gpu_common.cuh"
#include "logger.h"
#include "params.h"
#include "preprocess_cu.h"
#include "decode.h"

namespace ParkingPerception
{
namespace MultiTaskDet
{
struct NvInferDeleter
{
  template <typename T>
  void operator()(T* obj) const
  {
    // if (obj)
    // {
    //   obj->destroy();
    // }
  }
};

struct InferResults
{
  InferResults(){};
  std::vector<std::vector<float>> det_result;  //{left, top, right, bottom, label, confidence, x1, y1, x2, y2,theta, id}
  cv::Mat da_result;                           //{0:Background, 1:Foreground}
  cv::Mat ll_result;                           //{0:Background, 1:Parking_space_line, 2:Others}
  InferResults operator=(const InferResults& other)
  {
    this->det_result = other.det_result;
    this->da_result = other.da_result.clone();
    this->ll_result = other.ll_result.clone();

    return *this;
  }
};

class AVM_MultiTaskDet
{
public:
  AVM_MultiTaskDet(std::string config_path);
  int init();
  int inference(const cv::Mat& input_img);
  InferResults* get_results();

private:
  int load_config();
  void preprocess(uint8_t* img_device, float* buffer, int batch, int width, int height, cudaStream_t stream);
  void decode();

private:
  std::string config_path_;
  Params infer_params_;  //推理输入参数
  // cuda and trt
  cudaStream_t stream_;                                                   // stream
  std::unique_ptr<nvinfer1::IExecutionContext, NvInferDeleter> context_;  //网络上下文
  std::vector<std::string> binding_names_;                                //存储网络节点名称
  std::map<std::string, std::pair<int, size_t>> engine_name_size_;        //存储网络节点名及对应序号与size
  std::unique_ptr<GPUAllocator> gpu_mem_;                                 // gpu内存分配
  float* buffers_[32];                                                    // buffer内存地址
  uint8_t* pdst_device_;                                                  //输入数据gpu地址（cuda预处理用）
  float* host_input_;                                                     //输入数据cpu地址
  float* host_det_out_;                                                   //检测头输出数据cpu地址
  int* host_da_seg_;                                                      //可行驶区域分割输出数据cpu地址
  int* host_ll_seg_;                                                      //地面标线分割输出数据cpu地址
  std::vector<size_t> size_data_;                                         //输入输出size
  InferResults infer_results_;                                            //检测结果
};
}  // namespace MultiTaskDet
}  // namespace ParkingPerception
