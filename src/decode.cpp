#include "decode.h"

namespace ParkingPerception
{
namespace MultiTaskDet
{
void get_det_result(std::vector<std::vector<float>>& box_result, float* tensor, float confidence_threshold,
                    float nms_threshold)
{
  auto data = reinterpret_cast<float*>(tensor);
  //确保输出正确
  int obj_num = 3 * (100 * 100 + 50 * 50 + 25 * 25);  //输出目标总数量
  int data_num = 14;  // 每个目标包含的数据,x,y,w,h,obj_conf,cls_conf_0,cls_conf_1,x1,y1,x2,y2,theta_0,theta_1,theta_2

  // decode box,batch默认为1
  std::vector<std::vector<float>> bboxes;
  for (auto i = 0; i < obj_num; i++)  //遍历object
  {
    //当前obj的首地址
    float* ptr = data + i * data_num;

    // obj_conf
    float objness = ptr[4];
    //取出达到置信度阈值的结果
    if (objness < confidence_threshold)
      continue;

    //二分类获得最大类别索引
    std::vector<float> cls_scores = { ptr[5], ptr[6] };
    int label = std::max_element(cls_scores.begin(), cls_scores.end()) - cls_scores.begin();
    float confidence = cls_scores[label] * objness;  //置信度
    //再次验证置信度满足要求
    if (confidence < confidence_threshold)
      continue;

    // 中心点、宽、高
    float cx = ptr[0];
    float cy = ptr[1];
    float width = ptr[2];
    float height = ptr[3];

    // 左上右下x,y,对应网络输入图的位置
    float left = cx - width * 0.5;
    float top = cy - height * 0.5;
    float right = cx + width * 0.5;
    float bottom = cy + height * 0.5;

    //关键点x,y，对应网络输入图的位置
    float x1 = ptr[7];
    float y1 = ptr[8];
    float x2 = ptr[9];
    float y2 = ptr[10];

    //求解角度
    std::vector<float> theta_list = { ptr[12], ptr[13], ptr[11] };
    float sum_0 = 0;
    float sum_1 = 0;
    for (int i = 0; i < 3; i++)
    {
      sum_0 += theta_list[i] * std::sin(2 * (i + 1) * M_PI / 3.0);
      sum_1 += theta_list[i] * std::cos(2 * (i + 1) * M_PI / 3.0);
    }

    float theta = -std::atan2(sum_0, sum_1);  // 值域为[-pi,pi]

    bboxes.push_back({ left, top, right, bottom, float(label), confidence, x1, y1, x2, y2, theta, -1 });
  }
  // printf("decoded bboxes.size = %d\n", bboxes.size());

  // nms非极大抑制
  std::sort(bboxes.begin(), bboxes.end(), [](std::vector<float>& a, std::vector<float>& b) { return a[5] > b[5]; });
  std::vector<bool> remove_flags(bboxes.size());
  // std::vector<std::vector<float>> box_result;
  box_result.clear();
  box_result.reserve(bboxes.size());

  auto iou = [](const std::vector<float>& a, const std::vector<float>& b) {
    float cross_left = std::max(a[0], b[0]);
    float cross_top = std::max(a[1], b[1]);
    float cross_right = std::min(a[2], b[2]);
    float cross_bottom = std::min(a[3], b[3]);

    float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
    float union_area = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]) +
                       std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]) - cross_area;
    if (cross_area == 0 || union_area == 0)
      return 0.0f;
    return cross_area / union_area;
  };

  for (int i = 0; i < bboxes.size(); ++i)
  {
    if (remove_flags[i])
      continue;

    auto& ibox = bboxes[i];
    box_result.emplace_back(ibox);
    for (int j = i + 1; j < bboxes.size(); ++j)
    {
      if (remove_flags[j])
        continue;

      auto& jbox = bboxes[j];
      if (ibox[4] == jbox[4])
      {
        // class matched
        if (iou(ibox, jbox) >= nms_threshold)
          remove_flags[j] = true;
      }
    }
  }
  // printf("box_result.size = %d\n", box_result.size());
}

void get_seg_result(cv::Mat& output, int* tensor)
{
  auto data = reinterpret_cast<uint32_t*>(tensor);
  uint8_t* output_ptr = output.ptr<uint8_t>();
  for (int hh = 0; hh < output.rows; ++hh)
  {
    for (int ww = 0; ww < output.cols; ++ww)
    {
      *output_ptr++ = uint8_t(*data++);
    }
  }
}

}  // namespace MultiTaskDet
}  // namespace ParkingPerception