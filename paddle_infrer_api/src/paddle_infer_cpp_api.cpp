#include <iostream>
#include <numeric>
#include <time.h>
#include "paddle_inference_api.h"
#include "opencv2/opencv.hpp"
#include<windows.h>

// 将wchar_t*字符串指针转换为string字符串格式
std::string wchar_to_string(const wchar_t* wchar);
// 将string字符串指针转换为wchar_t*字符串格式
wchar_t* string_to_wchar(const std::string str);
// 将图片的矩阵数据转换为opencv的mat数据
cv::Mat data_to_mat(uchar* data, size_t size);
// 将图片的矩阵数据转换为opencv的mat数据
std::vector<float> input_data_process(std::vector<cv::Mat> input_image, std::vector<int> shape,
    int type);
// 构建放射变换矩阵
cv::Mat get_affine_transform(cv::Point center, cv::Size input_size, int rot, cv::Size output_size,
    cv::Point2f shift = cv::Point2f(0, 0));



// @brief 将wchar_t*字符串指针转换为string字符串格式
// @param wchar 输入字符指针
// @return 转换出的string字符串 
std::string wchar_to_string(const wchar_t* wchar) {
    // 获取输入指针的长度
    int path_size = WideCharToMultiByte(CP_OEMCP, 0, wchar, wcslen(wchar), NULL, 0, NULL, NULL);
    char* chars = new char[path_size + 1];
    // 将双字节字符串转换成单字节字符串
    WideCharToMultiByte(CP_OEMCP, 0, wchar, wcslen(wchar), chars, path_size, NULL, NULL);
    chars[path_size] = '\0';
    std::string pattern = chars;
    delete chars; //释放内存
    return pattern;
}

// @brief 将string字符串指针转换为wchar_t*字符串格式
// @param str 输入字符串
// @return 转换出的wchar_t* 字符指针
wchar_t* string_to_wchar(const std::string str)
{
    //string 转 char*
    const char* chars = str.c_str();
    //第一次调用返回转换后的字符串长度，用于确认为wchar_t*开辟多大的内存空间
    int size = MultiByteToWideChar(CP_OEMCP, 0, chars, strlen(chars) + 1, NULL, 0);
    wchar_t* wchar = new wchar_t[size];
    //第二次调用将单字节字符串转换成双字节字符串
    MultiByteToWideChar(CP_OEMCP, 0, chars, strlen(chars) + 1, wchar, size);
    return wchar;
}

// @brief 将图片的矩阵数据转换为opencv的mat数据
// @param data 图片矩阵
// @param size 图片矩阵长度
// @return 转换后的mat数据
cv::Mat data_to_mat(uchar* data, size_t size) {
    //将图片数组数据读取到容器中
    std::vector<uchar> buf;
    for (int i = 0; i < size; i++) {
        buf.push_back(*data);
        data++;
    }
    // 利用图片解码，将容器中的数据转换为mat类型
    return cv::imdecode(cv::Mat(buf), 1);
}

// @brief 将图片的矩阵数据转换为opencv的mat数据
// @param input_image 输入图片数据
// @param shape 模型输入形状
// @param type 数据处理方式
// @return 处理完的输入数据
std::vector<float> input_data_process(std::vector<cv::Mat> input_image, std::vector<int> shape,
    int type) {

    int width = shape[3]; // 要求输入图片数据的宽度
    int height = shape[2]; // 要求输入图片数据的高度
    int channels = shape[1]; // 要求输入图片数据的维度
    int bath_size = shape[0]; // 要求输入的bath_size
    std::cout << "width  " << width << std::endl;
    std::cout << "height  " << height << std::endl;
    std::cout << "channels  " << channels << std::endl;
    std::cout << "bath_size  " << bath_size << std::endl;
    std::vector<float> input_data(bath_size * channels * height * width,1);
    for (int b = 0; b < bath_size; b++) {
        cv::Mat blob_image;
        cv::cvtColor(input_image[b], blob_image, cv::COLOR_BGR2RGB); // 将图片通道由 BGR 转为 RGB

        if (type == 0) {
            // 对输入图片按照tensor输入要求进行缩放
            cv::resize(blob_image, blob_image, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
            // 图像数据归一化，减均值mean，除以方差std
            // PaddleDetection模型使用imagenet数据集的均值 Mean = [0.485, 0.456, 0.406]和方差 std = [0.229, 0.224, 0.225]
            std::vector<float> mean_values{ 0.485 * 255, 0.456 * 255, 0.406 * 255 };
            std::vector<float> std_values{ 0.229 * 255, 0.224 * 255, 0.225 * 255 };
            std::vector<cv::Mat> rgb_channels(3);
            cv::split(blob_image, rgb_channels); // 分离图片数据通道
            for (auto i = 0; i < rgb_channels.size(); i++) {
                //分通道依此对每一个通道数据进行归一化处理
                rgb_channels[i].convertTo(rgb_channels[i], CV_32FC1, 1.0 / std_values[i], (0.0 - mean_values[i]) / std_values[i]);
            }
            cv::merge(rgb_channels, blob_image); // 合并图片数据通道
        }
        else if (type == 1) {
            // 对输入图片按照tensor输入要求进行缩放
            cv::resize(blob_image, blob_image, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
            // 图像数据归一化
            std::vector<float> std_values{ 255.0,  255.0,  255.0 };
            std::vector<cv::Mat> rgb_channels(3);
            cv::split(blob_image, rgb_channels); // 分离图片数据通道
            for (auto i = 0; i < rgb_channels.size(); i++) {
                //分通道依此对每一个通道数据进行归一化处理
                rgb_channels[i].convertTo(rgb_channels[i], CV_32FC1, 1.0 / std_values[i]);
            }
            cv::merge(rgb_channels, blob_image); // 合并图片数据通道
        }
        else if (type == 2) {
            // 获取仿射变换信息
            cv::Point center(blob_image.cols / 2, blob_image.rows / 2); // 变换中心
            cv::Size input_size(blob_image.cols, blob_image.rows); // 输入尺寸
            int rot = 0; // 角度
            cv::Size output_size(width, height); // 输出尺寸

            // 获取仿射变换矩阵
            cv::Mat warp_mat(2, 3, CV_32FC1);
            warp_mat = get_affine_transform(center, input_size, rot, output_size);
            // 仿射变化
            cv::warpAffine(blob_image, blob_image, warp_mat, output_size, cv::INTER_LINEAR);
            // 图像数据归一化
            std::vector<float> mean_values{ 0.485 * 255, 0.456 * 255, 0.406 * 255 };
            std::vector<float> std_values{ 0.229 * 255, 0.224 * 255, 0.225 * 255 };
            std::vector<cv::Mat> rgb_channels(3);
            cv::split(blob_image, rgb_channels); // 分离图片数据通道
            for (auto i = 0; i < rgb_channels.size(); i++) {
                //分通道依此对每一个通道数据进行归一化处理
                rgb_channels[i].convertTo(rgb_channels[i], CV_32FC1, 1.0 / std_values[i], (0.0 - mean_values[i]) / std_values[i]);
            }
            cv::merge(rgb_channels, blob_image); // 合并图片数据通道
        }

        // 原有图片数据为 H、W、C 格式，输入要求的为 C、H、W 格式
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    input_data[b * width * height * channels + c * width * height + h * width + w] 
                        = blob_image.at<cv::Vec<float, 3>>(h, w)[c];
                }
            }
        }
    
    }
    return input_data;
}


// @brief 构建放射变换矩阵
// @param center 中心点
// @param input_size 输入尺寸
// @param rot 角度
// @param output_size 输出尺寸
// @param shift 
// @rrturn 变换矩阵
cv::Mat get_affine_transform(cv::Point center, cv::Size input_size, int rot, cv::Size output_size,
    cv::Point2f shift) {

    // 输入尺寸宽度
    int src_w = input_size.width;

    // 输出尺寸
    int dst_w = output_size.width;
    int dst_h = output_size.height;

    // 旋转角度
    float rot_rad = 3.1715926f * rot / 180.0;
    int pt = (int)src_w * -0.5;
    float sn = std::sin(rot_rad);
    float cs = std::cos(rot_rad);

    cv::Point2f src_dir(-1.0 * pt * sn, pt * cs);
    cv::Point2f dst_dir(0.0, dst_w * -0.5);
    // 输入三个点
    cv::Point2f src[3];
    src[0] = cv::Point2f(center.x + input_size.width * shift.x, center.y + input_size.height * shift.y);
    src[1] = cv::Point2f(center.x + src_dir.x + input_size.width * shift.x, center.y + src_dir.y + input_size.height * shift.y);
    cv::Point2f direction = src[0] - src[1];
    src[2] = cv::Point2f(src[1].x - direction.y, src[1].y - direction.x);
    // 输出三个点
    cv::Point2f dst[3];
    dst[0] = cv::Point2f(dst_w * 0.5, dst_h * 0.5);
    dst[1] = cv::Point2f(dst_w * 0.5 + dst_dir.x, dst_h * 0.5 + dst_dir.y);
    direction = dst[0] - dst[1];
    dst[2] = cv::Point2f(dst[1].x - direction.y, dst[1].y - direction.x);

    return cv::getAffineTransform(src, dst);

}


typedef struct paddle_inference {
    paddle_infer::Config config;
    std::shared_ptr<paddle_infer::Predictor> predictor;
}PaddleInfer;


// @brief 初始化PaddleInfer推理核心
// @return 返回PaddleInfer结构体指针
extern "C" __declspec(dllexport) void* __stdcall paddle_infer_init() {
    PaddleInfer* paddle_infer = new PaddleInfer(); // 初始化
    return (void*)paddle_infer;
}

// @brief 加载本地Paddle模型
// @param paddle_infer_ptr PaddleInfer结构体指针
// @param model_path_wchar 网络结构的文件路径
// @param params_path_wchar 模型参数的文件路径
// @return 返回PaddleInfer结构体指针
extern "C" __declspec(dllexport) void* __stdcall set_model(void* paddle_infer_ptr, 
    const wchar_t* model_path_wchar, const wchar_t* params_path_wchar) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    //读取接口输入参数
    std::string model_path = wchar_to_string(model_path_wchar);// 网络结构的文件路径
    std::string params_path = wchar_to_string(params_path_wchar);// 模型参数的文件路径
    if (model_path == " ") {
        paddle_infer->config.SetModel(model_path); // 只加载模型参数的文件路径
    }
    else {
        paddle_infer->config.SetModel(model_path, params_path);
    }
    return (void*)paddle_infer;
}

// @brief 设置模型推理方式为CPU推理
// @param paddle_infer_ptr PaddleInfer结构体指针
// @param cpu_math_library_num_threads CPUBlas库线程数
// @return 返回PaddleInfer结构体指针
extern "C" __declspec(dllexport) void* __stdcall predictor_device_cpu(void* paddle_infer_ptr,
    int cpu_math_library_num_threads) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    paddle_infer->config.DisableGpu(); // 忽略GPU，使用CPU
    paddle_infer->config.SetCpuMathLibraryNumThreads(cpu_math_library_num_threads); // 设置CPUBlas库线程数
    // 读取配置创建预测器
    paddle_infer->predictor = paddle_infer::CreatePredictor(paddle_infer->config);
    return (void*)paddle_infer;
}

// @brief 设置推理方式为GPU推理
// @param paddle_infer_ptr PaddleInfer结构体指针
// @param memory_init_size 初始化使用显存大小，单位MB
// @param device_id 显卡编号，默认为0
// @param workspace_size 工作显存大小
// @return 返回PaddleInfer结构体指针
extern "C" __declspec(dllexport) void* __stdcall predictor_device_gpu(void* paddle_infer_ptr,
    unsigned long long memory_init_size, int device_id, int workspace_size) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    paddle_infer->config.EnableUseGpu(memory_init_size, device_id); // 开启GPU，使用GPU加速
    paddle_infer->config.SwitchIrOptim(true); // 执行IR图优化
    paddle_infer->config.EnableMemoryOptim(); // 打开内存优化
    paddle_infer->config.EnableTensorRtEngine(static_cast<int64_t>(1 << workspace_size), 1, 10,
        paddle::AnalysisConfig::Precision::kFloat32, false, false); // 打开TensorRT引擎
    // 读取配置创建预测器
    paddle_infer->predictor = paddle_infer::CreatePredictor(paddle_infer->config);
    return (void*)paddle_infer;
}


// @brief 设置推理方式为ONNX runtime
// @param paddle_infer_ptr PaddleInfer结构体指针
// @param cpu_math_library_num_threads ONNXRuntime算子计算线程数
// @return 返回PaddleInfer结构体指针
extern "C" __declspec(dllexport) void* __stdcall predictor_device_ONNX_runtime(void* paddle_infer_ptr,
    int cpu_math_library_num_threads) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    paddle_infer->config.EnableONNXRuntime(); // 启用 ONNXRuntime
    paddle_infer->config.EnableORTOptimization();  // 开启ONNXRuntime优化
    paddle_infer->config.SetCpuMathLibraryNumThreads(10); // 设置 ONNXRuntime 算子计算线程数
    paddle_infer->config.DisableONNXRuntime(); // 禁用 ONNXRuntime 进行预测
    // 读取配置创建预测器
    paddle_infer->predictor = paddle_infer::CreatePredictor(paddle_infer->config);
    return (void*)paddle_infer;
}

// @brief 设置推理方式为oneDNN
// @param paddle_infer_ptr PaddleInfer结构体指针
// @param capacity oneDNN的cache数量
// @return 返回PaddleInfer结构体指针
extern "C" __declspec(dllexport) void* __stdcall predictor_device_oneDNN(void* paddle_infer_ptr,
    int capacity) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    paddle_infer->config.EnableMKLDNN(); // 启用 oneDNN 进行预测
    // 设置 oneDNN 的 cache 数量
    // 当动态shape推理时，能缓存n个最新输入shape对应的oneDNN配置，减少shape变换时重新生成配置带来的开销
    paddle_infer->config.SetMkldnnCacheCapacity(capacity);
    // 读取配置创建预测器
    paddle_infer->predictor = paddle_infer::CreatePredictor(paddle_infer->config);
    return (void*)paddle_infer;
}

// @brief 获取输入节点名称
// @param paddle_infer_ptr PaddleInfer结构体指针
// @param length 字符串长度
// @return 输入节点字符串
extern "C" __declspec(dllexport) wchar_t* __stdcall get_input_names(void* paddle_infer_ptr,
    int* length) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    // 获取节点名称
    std::vector<std::string> input_names = paddle_infer->predictor->GetInputNames();
    // 将节点转为可以传递的格式
    std::string out_str;
    for (int i = 0; i < input_names.size(); i++) {
        out_str += input_names[i];
        length[i] = input_names[i].length();
    }
    return string_to_wchar(out_str);
}


// @brief 设置输入形状
// @param paddle_infer_ptr PaddleInfer结构体指针
// @param input_name_wchar 输入节点名称
// @param input_shape 输入形状
// @return 返回PaddleInfer结构体指针
extern "C" __declspec(dllexport) void* __stdcall set_input_shape(void* paddle_infer_ptr,
    wchar_t* input_name_wchar, int* input_shape, int length) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    std::string input_name = wchar_to_string(input_name_wchar);
    std::vector<int> shape(input_shape, input_shape + length);
    std::unique_ptr<paddle_infer::Tensor> input_tensor = paddle_infer->predictor->GetInputHandle(input_name);
    input_tensor->Reshape(shape);
    return (void*)paddle_infer;
}

// @brief 加载图片数据
// @param paddle_infer_ptr PaddleInfer结构体指针
// @param input_name_wchar 输入节点名称
// @param image_data 图片数据
// @param image_size 图片数据长度
// @param type 数据处理方式
// @return 返回PaddleInfer结构体指针
extern "C" __declspec(dllexport) void* __stdcall load_input_image_data(void* paddle_infer_ptr,
    wchar_t* input_name_wchar, uchar* image_data, size_t image_size, int type) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    std::string input_name = wchar_to_string(input_name_wchar);
    cv::Mat input_image = data_to_mat(image_data, image_size); // 解码输入图片
    // 获取输入Tensor
    std::unique_ptr<paddle_infer::Tensor> input_tensor = paddle_infer->predictor->GetInputHandle(input_name);
    std::vector<int> input_shape = input_tensor->shape(); // 获取输入Tensor形状
    // 获取输入数据
    std::vector<cv::Mat> input_images;
    input_images.push_back(input_image);
    std::vector<float> input_datas = input_data_process(input_images, input_shape, type); // 处理输入数据
    // 加载数据
    input_tensor->CopyFromCpu(input_datas.data());
    return (void*)paddle_infer;
}

// @brief 加载普通数据
// @param paddle_infer_ptr PaddleInfer结构体指针
// @param input_name_wchar 输入节点名称
// @param data 输入数据
// @return 返回PaddleInfer结构体指针
extern "C" __declspec(dllexport) void* __stdcall load_input_data(void* paddle_infer_ptr,
    wchar_t* input_name_wchar, float* data) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    std::string input_name = wchar_to_string(input_name_wchar);
    // 获取输入Tensor
    std::unique_ptr<paddle_infer::Tensor> input_tensor = paddle_infer->predictor->GetInputHandle(input_name);
    std::vector<int> input_shape = input_tensor->shape(); // 获取输入Tensor形状
    // 获取输入长度
    int data_length = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>());
    // 构建输入数据
    std::vector<float> input_datas(data_length,1);
    for (int i = 0; i < data_length; i++) {
        input_datas.push_back(data[i]);
    }
    input_tensor->CopyFromCpu(input_datas.data()); // 加载输入数据
    return (void*)paddle_infer;
}


// @brief 模型推理
// @param paddle_infer_ptr PaddleInfer结构体指针
// @return 返回PaddleInfer结构体指针
extern "C" __declspec(dllexport) void* __stdcall infer(void* paddle_infer_ptr) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    paddle_infer->predictor->Run();
    return (void*)paddle_infer;
}


// @brief 获取输出节点名称
// @param paddle_infer_ptr PaddleInfer结构体指针
// @param length 字符串长度
// @return 输出节点字符串
extern "C" __declspec(dllexport) wchar_t* __stdcall get_output_names(void* paddle_infer_ptr,
    int* length) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    // 获取节点名称
    std::vector<std::string> input_names = paddle_infer->predictor->GetOutputNames();
    // 将节点转为可以传递的格式
    std::string out_str;
    for (int i = 0; i < input_names.size(); i++) {
        out_str += input_names[i];
        length[i] = input_names[i].length();
    }
    return string_to_wchar(out_str);
}

// @brief 获取指定节点的形状
// @param paddle_infer_ptr PaddleInfer结构体指针
// @param node_name_wchar 节点名
// @param shape 形状
// @param dimension 维度
// @return 输出结果
extern "C" __declspec(dllexport) void __stdcall get_node_shape(void* paddle_infer_ptr,
    wchar_t* node_name_wchar, int* shape, int* dimension) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    std::string name = wchar_to_string(node_name_wchar);
    std::unique_ptr<paddle_infer::Tensor> tensor = paddle_infer->predictor->GetOutputHandle(name);
    std::vector<int> shapes = tensor->shape();
    for (int i = 0; i < shapes.size(); i++) {
        *shape = shapes[i];
        shape++;
    }
    *dimension = shapes.size();
}

// @brief 读取模型结果输出-F32
// @param paddle_infer_ptr PaddleInfer结构体指针
// @param output_name_wchar 输出节点名
// @return 输出结果
extern "C" __declspec(dllexport) void __stdcall read_result_data_F32(void* paddle_infer_ptr,
    wchar_t * output_name_wchar, float* infer_result) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    std::string output_name = wchar_to_string(output_name_wchar);
    // 获取输出节点句柄
    std::unique_ptr<paddle_infer::Tensor> output_tensor = paddle_infer->predictor->GetOutputHandle(output_name);
    std::vector<int> output_shape = output_tensor->shape();
    int data_length = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    std::vector<float> result(data_length);
    output_tensor->CopyToCpu(result.data()); // 读取结果
    for (int i = 0; i < data_length; i++) {
        *infer_result = result[i];
        infer_result++;
    }
}

// @brief 读取模型结果输出-I32
// @param paddle_infer_ptr PaddleInfer结构体指针
// @param output_name_wchar 输出节点名
// @return 输出结果
extern "C" __declspec(dllexport) void __stdcall read_result_data_I32(void* paddle_infer_ptr,
    wchar_t* output_name_wchar, int* infer_result) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    std::string output_name = wchar_to_string(output_name_wchar);
    // 获取输出节点句柄
    std::unique_ptr<paddle_infer::Tensor> output_tensor = paddle_infer->predictor->GetOutputHandle(output_name);
    std::vector<int> output_shape = output_tensor->shape();
    int data_length = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    std::vector<int> result(data_length);
    output_tensor->CopyToCpu(result.data()); // 读取结果
    for (int i = 0; i < data_length; i++) {
        *infer_result = result[i];
        infer_result++;
    }
}
// @brief 读取模型结果输出-I64
// @param paddle_infer_ptr PaddleInfer结构体指针
// @param output_name_wchar 输出节点名
// @return 输出结果
extern "C" __declspec(dllexport) void __stdcall read_result_data_I64(void* paddle_infer_ptr,
    wchar_t* output_name_wchar, long long* infer_result) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    std::string output_name = wchar_to_string(output_name_wchar);
    // 获取输出节点句柄
    std::unique_ptr<paddle_infer::Tensor> output_tensor = paddle_infer->predictor->GetOutputHandle(output_name);
    std::vector<int> output_shape = output_tensor->shape();
    int data_length = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    std::vector<long long> result(data_length);
    output_tensor->CopyToCpu(result.data()); // 读取结果
    for (int i = 0; i < data_length; i++) {
        *infer_result = result[i];
        infer_result++;
    }
}



// @brief 销毁内存
// @param paddle_infer_ptr PaddleInfer结构体指针
extern "C" __declspec(dllexport) void __stdcall dispose(void* paddle_infer_ptr) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    delete paddle_infer;
}


