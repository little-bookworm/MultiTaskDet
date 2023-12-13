/*
 * @Author: zjj
 * @Date: 2023-12-04 16:02:47
 * @LastEditors: zjj
 * @LastEditTime: 2023-12-13 13:39:10
 * @FilePath: /MultiTaskDet/test/test.cpp
 * @Description:
 *
 * Copyright (c) 2023 by ${git_name_email}, All Rights Reserved.
 */
#include <string>
#include <opencv2/opencv.hpp>

#include "inference_api.h"

using namespace ParkingPerception::MultiTaskDet;

static std::vector<cv::Scalar> _parking_color = { { 0, 255, 0 }, { 0, 0, 255 } };
static std::vector<std::vector<int>> _color = { { 0, 0, 0 }, { 0, 255, 0 }, { 0, 0, 255 }, { 0, 255, 255 } };

void get_parking_corners(const cv::Point2f pt0, const cv::Point2f pt1, const float depth, const float theta,
                         std::vector<cv::Point2f>& corners)
{
  float delta_x = depth * std::cos(theta);
  float delta_y = depth * std::sin(theta);
  cv::Point2f pt3 = { pt0.x - delta_x, pt0.y - delta_y };
  cv::Point2f pt2 = { pt1.x - delta_x, pt1.y - delta_y };

  //依次存入车位0，1，2，3点及中心点
  corners.clear();
  corners.push_back(pt0);
  corners.push_back(pt1);
  corners.push_back(pt2);
  corners.push_back(pt3);
}

void render(cv::Mat& image, const cv::Mat& iclass, const std::string flag)
{
  auto pimage = image.ptr<cv::Vec3b>(0);
  // auto pprob = prob.ptr<float>(0);
  auto pclass = iclass.ptr<uint8_t>(0);

  // for (int i = 0; i < image.cols * image.rows; ++i, ++pimage, ++pprob, ++pclass)
  for (int i = 0; i < image.cols * image.rows; ++i, ++pimage, ++pclass)
  {
    int iclass = *pclass;
    // float probability = *pprob;
    auto& pixel = *pimage;
    float foreground;
    if (iclass == 0)
    {
      foreground = 0.0;
    }
    else
    {
      foreground = 0.8;
    }
    // float foreground = std::min(0.6f + probability * 0.2f, 0.5f);
    float background = 1 - foreground;
    for (int c = 0; c < 3; ++c)
    {
      float value;
      if (flag == "da")
      {
        value = pixel[c] * background + foreground * _color[iclass][c];
      }
      else if (flag == "ll")
      {
        value = pixel[c] * background + foreground * _color[iclass + 1][c];
      }
      pixel[c] = std::min((int)value, 255);
    }
  }
}

int main()
{
  // prepare input
  cv::Mat img_raw = cv::imread("/hostdata/projects/parking_perception/modules/MultiTaskDet/test/test.jpg");
  cv::Mat img_input;
  cv::resize(img_raw, img_input, cv::Size(800, 800));

  //实例化推理模型
  std::string config_file = "/hostdata/projects/parking_perception/modules/MultiTaskDet/config/MultiTaskDet.yaml";
  AVM_MultiTaskDet* det = CreateDetection(config_file);

  //模型初始化
  if (0 != det->init())
  {
    std::cout << "Init failed" << std::endl;
    return 0;
  }

  //模型推理
  if (0 != det->inference(img_input))
  {
    std::cout << "Infer failed" << std::endl;
    return 0;
  }

  //检测结果
  InferResults* rlt = det->get_results();

  //绘制检测结果
  cv::Mat draw_det = img_input.clone();
  for (auto& det_rlt : rlt->det_result)
  {
    float left = det_rlt[0];
    float top = det_rlt[1];
    float right = det_rlt[2];
    float bottom = det_rlt[3];
    float label = det_rlt[4];
    float confidence = det_rlt[5];
    float x0 = det_rlt[6];
    float y0 = det_rlt[7];
    cv::Point2f pt0 = cv::Point2f(x0, y0);
    float x1 = det_rlt[8];
    float y1 = det_rlt[9];
    cv::Point2f pt1 = cv::Point2f(x1, y1);
    float theta = det_rlt[10];  //-pi~pi
    float id = det_rlt[11];     //-1

    //计算车位四个角点
    std::vector<cv::Point2f> corners;
    float depth = 300;  //根据车位先验给定
    get_parking_corners(pt0, pt1, depth, theta, corners);

    cv::line(draw_det, cv::Point(int(corners[0].x), int(corners[0].y)), cv::Point(int(corners[1].x), int(corners[1].y)),
             _parking_color[label], 2);
    cv::line(draw_det, cv::Point(int(corners[1].x), int(corners[1].y)), cv::Point(int(corners[2].x), int(corners[2].y)),
             _parking_color[label], 2);
    cv::line(draw_det, cv::Point(int(corners[2].x), int(corners[2].y)), cv::Point(int(corners[3].x), int(corners[3].y)),
             _parking_color[label], 2);
    cv::line(draw_det, cv::Point(int(corners[3].x), int(corners[3].y)), cv::Point(int(corners[0].x), int(corners[0].y)),
             _parking_color[label], 2);
  }

  //绘制da分割结果
  cv::Mat draw_da = img_input.clone();
  render(draw_da, rlt->da_result, "da");

  //绘制ll分割
  cv::Mat draw_ll = img_input.clone();
  render(draw_ll, rlt->ll_result, "ll");

  cv::imwrite("/hostdata/projects/parking_perception/modules/MultiTaskDet/test/out_det.jpg", draw_det);
  std::cout << "Save out_det.jpg." << std::endl;

  cv::imwrite("/hostdata/projects/parking_perception/modules/MultiTaskDet/test/out_da.jpg", draw_da);
  std::cout << "Save out_da.jpg." << std::endl;
  cv::imwrite("/hostdata/projects/parking_perception/modules/MultiTaskDet/test/da_mask.bmp", rlt->da_result);
  std::cout << "Save da_mask.bmp." << std::endl;

  cv::imwrite("/hostdata/projects/parking_perception/modules/MultiTaskDet/test/out_ll.jpg", draw_ll);
  std::cout << "Save out_ll.jpg." << std::endl;
  cv::imwrite("/hostdata/projects/parking_perception/modules/MultiTaskDet/test/ll_mask.bmp", rlt->ll_result);
  std::cout << "Save ll_mask.bmp." << std::endl;

  return 0;
}
