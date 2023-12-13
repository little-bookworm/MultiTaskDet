#include "inference.h"

namespace ParkingPerception
{
namespace MultiTaskDet
{
AVM_MultiTaskDet::AVM_MultiTaskDet(std::string infer_config_path)
{
  if (0 != load_config(infer_config_path))
  {
    std::cout << "[MultiTaskDet]->[constructor] Failed to load config file." << std::endl;
    return;
  }
  std::cout << "[MultiTaskDet]->[constructor] Loading config file success." << std::endl;
}

int AVM_MultiTaskDet::init()
{
  // 设置device
  CHECK_CUDA(cudaSetDevice(infer_params_.device_id));
  // 设置stream;
  CHECK_CUDA(cudaStreamCreate(&stream_));
  // 从配置文件读取model
  std::vector<char> trtModelStream_;
  size_t model_size(0);
  std::ifstream file(infer_params_.model_file, std::ios::binary);
  if (file.good())
  {
    file.seekg(0, file.end);
    model_size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream_.resize(model_size);
    file.read(trtModelStream_.data(), model_size);
    file.close();
  }
  else
  {
    std::cout << "[MultiTaskDet]->[init] Failed to Load engine file: " << infer_params_.model_file << std::endl;
    return -1;
  }
  //构建推理runtime
  auto runtime = std::unique_ptr<nvinfer1::IRuntime, NvInferDeleter>(nvinfer1::createInferRuntime(gLogger));
  if (!runtime)
  {
    std::cout << "[MultiTaskDet]->[init] Failed to create runtime from model file: " << infer_params_.model_file
              << std::endl;
    return -1;
  }
  //生成tensorrt引擎
  auto engine = std::unique_ptr<nvinfer1::ICudaEngine, NvInferDeleter>(
      runtime->deserializeCudaEngine(trtModelStream_.data(), model_size, nullptr));
  if (!engine)
  {
    std::cout << "[MultiTaskDet]->[init] Failed to create engine from model file: " << infer_params_.model_file
              << std::endl;
    return -1;
  }
  //创建上下文
  context_ = std::unique_ptr<nvinfer1::IExecutionContext, NvInferDeleter>(engine->createExecutionContext());
  if (!context_)
  {
    std::cout << "[MultiTaskDet]->[init] Failed to create context from model file: " << infer_params_.model_file
              << std::endl;
    return -1;
  }
  // 解析输入输出维度
  const int nb_bindings = engine->getNbBindings();  //输入输出节点数
  if (nb_bindings != infer_params_.tensorNum)
  {
    std::cout << "[MultiTaskDet]->[init] Tensor nb_bindings error !!!" << std::endl;
    return -1;
  }
  size_data_.clear();
  //遍历输入输出，获得tensor的name
  for (int i = 0; i < nb_bindings; ++i)
  {
    std::string name = context_->getEngine().getBindingName(i);  //获得输入输出tensor的name
    int index = find(infer_params_.tensorNames.begin(), infer_params_.tensorNames.end(), name) -
                infer_params_.tensorNames.begin();  //获取name在入参中对应的位置
    if (name != infer_params_.tensorNames[index])   //确保name一致
    {
      std::cout << "[MultiTaskDet]->[init] Tensor name error: " << infer_params_.tensorNames[index] << std::endl;
      return false;
    }

    auto dim = context_->getEngine().getBindingDimensions(i);  //获得输入输出tensor的维度
    if ((dim.d[0] != infer_params_.tensorDim0[index]) || (dim.d[1] != infer_params_.tensorDim1[index]) ||
        (dim.d[2] != infer_params_.tensorDim2[index]) ||
        (dim.d[3] != infer_params_.tensorDim3[index]))  //输入输出维度与config中保持一致
    {
      std::cout << "[MultiTaskDet]->[init] Tensor shape error !!!" << std::endl;
      return false;
    }

    size_t size = dim.d[0] * dim.d[1] * dim.d[2] * (dim.d[3] > 0 ? dim.d[3] : 1);  //数据数量
    size_data_.push_back(size);

    // std::cout << "i=" << i << " tensor's name:" << name << " dim:(" << dim.d[0] << "," << dim.d[1] << "," << dim.d[2]
    //           << "," << dim.d[3] << ")"
    //           << " size:" << size_data_[i] << std::endl;

    auto data_type = context_->getEngine().getBindingDataType(i);
    // if (data_type == nvinfer1::DataType::kFLOAT)
    // {
    //   std::cout << "type: kFLOAT" << std::endl;
    // }
    // else if (data_type == nvinfer1::DataType::kHALF)
    // {
    //   std::cout << "type: kHALF" << std::endl;
    // }
    // else if (data_type == nvinfer1::DataType::kINT8)
    // {
    //   std::cout << "type: kINT8" << std::endl;
    // }
    // else if (data_type == nvinfer1::DataType::kINT32)
    // {
    //   std::cout << "type: kINT32" << std::endl;
    // }
    // else if (data_type == nvinfer1::DataType::kBOOL)
    // {
    //   std::cout << "type: kBOOL" << std::endl;
    // }

    binding_names_.push_back(name);

    engine_name_size_.emplace(name, std::make_pair(i, size));  // name:(index,size)
  }

  //分配内存
  size_t size_img = infer_params_.img_w * infer_params_.img_h * 3 * 1;  // 输入3通道图像字节数
  CHECK_CUDA(
      cudaMalloc((void**)&pdst_device_, infer_params_.batch_size * size_img));  // GPU上开辟空间存放输入网络的bgr图像
  gpu_mem_.reset(new GPUAllocator);
  for (int i = 0; i < nb_bindings; ++i)
  {
    buffers_[i] = gpu_mem_->allocate<float>(
        binding_names_[i].c_str(),
        engine_name_size_[binding_names_[i]]
            .second);  // buffer内存分配(这里因为4个tensor不是float就是int32，大小一致，所以都用了float)
  }
  CHECK_CUDA(cudaMallocHost((void**)&host_input_, sizeof(float) * size_data_[0]));  // cpu上开辟输入数据内存
  CHECK_CUDA(cudaMallocHost((void**)&host_det_out_, sizeof(float) * size_data_[1]));  // cpu上开辟检测头输出数据内存
  CHECK_CUDA(cudaMallocHost((void**)&host_da_seg_, sizeof(int) * size_data_[2]));  // cpu上开辟分割头输出数据内存
  CHECK_CUDA(cudaMallocHost((void**)&host_ll_seg_, sizeof(int) * size_data_[3]));  // cpu上开辟分割头输出数据内存

  // decode数据分配内存
  infer_results_.da_result = cv::Mat(infer_params_.img_h, infer_params_.img_w, CV_8UC1);
  infer_results_.ll_result = cv::Mat(infer_params_.img_h, infer_params_.img_w, CV_8UC1);

  std::cout << "[MultiTaskDet]->[init] Infer init success." << std::endl;

  return 0;
}

int AVM_MultiTaskDet::inference(const cv::Mat& input_img)
{
  if (input_img.empty())
  {
    std::cout << "[MultiTaskDet]->[inference] Input_img is empty!!!" << std::endl;
    return -1;
  }

  // 数据预处理
  // 从cpu拷贝数据到gpu
  // cv::Mat input_img = input.clone();
  size_t size_img = infer_params_.img_w * infer_params_.img_h * 3 * 1;           //输入为uint8三通道图像
  CHECK_CUDA(cudaMemset(pdst_device_, 0, infer_params_.batch_size * size_img));  // gpu数据清0
  for (int index = 0; index < infer_params_.batch_size; index++)
  {
    CHECK_CUDA(cudaMemcpyAsync(pdst_device_ + index * infer_params_.img_w * infer_params_.img_h * 3, input_img.data,
                               size_img, cudaMemcpyHostToDevice, stream_));  // 将bgr图像从cpu复制到gpu
  }
  //流同步
  cudaStreamSynchronize(stream_);
  // 使用cuda进行预处理
  preprocess(pdst_device_, buffers_[engine_name_size_["images"].first], infer_params_.batch_size, infer_params_.img_w,
             infer_params_.img_h, stream_);
  //流同步
  cudaStreamSynchronize(stream_);

  // 推理
  context_->enqueue(infer_params_.batch_size, (void**)buffers_, stream_, nullptr);
  if (!context_)
  {
    std::cout << "[MultiTaskDet]->[inference] Failed to enqueue !" << std::endl;
    return -1;
  }

  // 将推理结果拷贝至cpu
  CHECK_CUDA(cudaMemcpyAsync(host_det_out_, buffers_[engine_name_size_["det_out"].first], sizeof(float) * size_data_[1],
                             cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA(cudaMemcpyAsync(host_da_seg_, buffers_[engine_name_size_["drive_area_seg"].first],
                             sizeof(float) * size_data_[2], cudaMemcpyDeviceToHost, stream_));
  CHECK_CUDA(cudaMemcpyAsync(host_ll_seg_, buffers_[engine_name_size_["lane_line_seg"].first],
                             sizeof(float) * size_data_[3], cudaMemcpyDeviceToHost, stream_));
  cudaStreamSynchronize(stream_);

  // 解码
  decode();

  return 0;
}

InferResults* AVM_MultiTaskDet::get_results()
{
  return &infer_results_;
}

int AVM_MultiTaskDet::load_config(std::string& config_path)
{
  //导入yaml文件
  YAML::Node infer_config;
  try
  {
    infer_config = YAML::LoadFile(config_path);
  }
  catch (const std::exception& e)
  {
    std::cout << "[MultiTaskDet]->[load_config] No config file: " << config_path << std::endl;
    return -1;
  }

  //导入配置参数
  auto img_params = infer_config["img_params"];
  infer_params_.batch_size = img_params["batch_size"].as<int>();
  infer_params_.img_h = img_params["img_h"].as<int>();
  infer_params_.img_w = img_params["img_w"].as<int>();
  auto model_params = infer_config["model_params"];
  infer_params_.model_file = model_params["model_path"].as<std::string>();
  infer_params_.confidence_threshold = model_params["confidence_threshold"].as<float>();
  infer_params_.nms_threshold = model_params["nms_threshold"].as<float>();
  infer_params_.tensorNum = model_params["tensorNum"].as<int>();
  infer_params_.tensorNames = model_params["tensorNames"].as<std::vector<std::string>>();
  infer_params_.tensorDim0 = model_params["tensorDim0"].as<std::vector<int>>();
  infer_params_.tensorDim1 = model_params["tensorDim1"].as<std::vector<int>>();
  infer_params_.tensorDim2 = model_params["tensorDim2"].as<std::vector<int>>();
  infer_params_.tensorDim3 = model_params["tensorDim3"].as<std::vector<int>>();
  auto device_params = infer_config["device_params"];
  infer_params_.device_id = device_params["device_id"].as<int>();

  return 0;
}

void AVM_MultiTaskDet::preprocess(uint8_t* img_device, float* buffer, int batch, int width, int height,
                                  cudaStream_t stream)
{
  preprocess_cu(img_device, buffer, batch, width, height, stream);
}

void AVM_MultiTaskDet::decode()
{
  //获得车位检测结果
  infer_results_.det_result.clear();
  get_det_result(infer_results_.det_result, host_det_out_, infer_params_.confidence_threshold,
                 infer_params_.nms_threshold);
  //获得可行驶区域分割结果
  get_seg_result(infer_results_.da_result, host_da_seg_);
  //获得车道线分割结果
  get_seg_result(infer_results_.ll_result, host_ll_seg_);
}
}  // namespace MultiTaskDet
}  // namespace ParkingPerception