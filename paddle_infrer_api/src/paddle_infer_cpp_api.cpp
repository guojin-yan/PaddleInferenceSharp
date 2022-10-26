#include <iostream>
#include <numeric>
#include <time.h>
#include "paddle_inference_api.h"
#include "opencv2/opencv.hpp"
#include<windows.h>

// ��wchar_t*�ַ���ָ��ת��Ϊstring�ַ�����ʽ
std::string wchar_to_string(const wchar_t* wchar);
// ��string�ַ���ָ��ת��Ϊwchar_t*�ַ�����ʽ
wchar_t* string_to_wchar(const std::string str);
// ��ͼƬ�ľ�������ת��Ϊopencv��mat����
cv::Mat data_to_mat(uchar* data, size_t size);
// ��ͼƬ�ľ�������ת��Ϊopencv��mat����
std::vector<float> input_data_process(std::vector<cv::Mat> input_image, std::vector<int> shape,
    int type);
// ��������任����
cv::Mat get_affine_transform(cv::Point center, cv::Size input_size, int rot, cv::Size output_size,
    cv::Point2f shift = cv::Point2f(0, 0));



// @brief ��wchar_t*�ַ���ָ��ת��Ϊstring�ַ�����ʽ
// @param wchar �����ַ�ָ��
// @return ת������string�ַ��� 
std::string wchar_to_string(const wchar_t* wchar) {
    // ��ȡ����ָ��ĳ���
    int path_size = WideCharToMultiByte(CP_OEMCP, 0, wchar, wcslen(wchar), NULL, 0, NULL, NULL);
    char* chars = new char[path_size + 1];
    // ��˫�ֽ��ַ���ת���ɵ��ֽ��ַ���
    WideCharToMultiByte(CP_OEMCP, 0, wchar, wcslen(wchar), chars, path_size, NULL, NULL);
    chars[path_size] = '\0';
    std::string pattern = chars;
    delete chars; //�ͷ��ڴ�
    return pattern;
}

// @brief ��string�ַ���ָ��ת��Ϊwchar_t*�ַ�����ʽ
// @param str �����ַ���
// @return ת������wchar_t* �ַ�ָ��
wchar_t* string_to_wchar(const std::string str)
{
    //string ת char*
    const char* chars = str.c_str();
    //��һ�ε��÷���ת������ַ������ȣ�����ȷ��Ϊwchar_t*���ٶ����ڴ�ռ�
    int size = MultiByteToWideChar(CP_OEMCP, 0, chars, strlen(chars) + 1, NULL, 0);
    wchar_t* wchar = new wchar_t[size];
    //�ڶ��ε��ý����ֽ��ַ���ת����˫�ֽ��ַ���
    MultiByteToWideChar(CP_OEMCP, 0, chars, strlen(chars) + 1, wchar, size);
    return wchar;
}

// @brief ��ͼƬ�ľ�������ת��Ϊopencv��mat����
// @param data ͼƬ����
// @param size ͼƬ���󳤶�
// @return ת�����mat����
cv::Mat data_to_mat(uchar* data, size_t size) {
    //��ͼƬ�������ݶ�ȡ��������
    std::vector<uchar> buf;
    for (int i = 0; i < size; i++) {
        buf.push_back(*data);
        data++;
    }
    // ����ͼƬ���룬�������е�����ת��Ϊmat����
    return cv::imdecode(cv::Mat(buf), 1);
}

// @brief ��ͼƬ�ľ�������ת��Ϊopencv��mat����
// @param input_image ����ͼƬ����
// @param shape ģ��������״
// @param type ���ݴ���ʽ
// @return ���������������
std::vector<float> input_data_process(std::vector<cv::Mat> input_image, std::vector<int> shape,
    int type) {

    int width = shape[3]; // Ҫ������ͼƬ���ݵĿ��
    int height = shape[2]; // Ҫ������ͼƬ���ݵĸ߶�
    int channels = shape[1]; // Ҫ������ͼƬ���ݵ�ά��
    int bath_size = shape[0]; // Ҫ�������bath_size
    std::cout << "width  " << width << std::endl;
    std::cout << "height  " << height << std::endl;
    std::cout << "channels  " << channels << std::endl;
    std::cout << "bath_size  " << bath_size << std::endl;
    std::vector<float> input_data(bath_size * channels * height * width,1);
    for (int b = 0; b < bath_size; b++) {
        cv::Mat blob_image;
        cv::cvtColor(input_image[b], blob_image, cv::COLOR_BGR2RGB); // ��ͼƬͨ���� BGR תΪ RGB

        if (type == 0) {
            // ������ͼƬ����tensor����Ҫ���������
            cv::resize(blob_image, blob_image, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
            // ͼ�����ݹ�һ��������ֵmean�����Է���std
            // PaddleDetectionģ��ʹ��imagenet���ݼ��ľ�ֵ Mean = [0.485, 0.456, 0.406]�ͷ��� std = [0.229, 0.224, 0.225]
            std::vector<float> mean_values{ 0.485 * 255, 0.456 * 255, 0.406 * 255 };
            std::vector<float> std_values{ 0.229 * 255, 0.224 * 255, 0.225 * 255 };
            std::vector<cv::Mat> rgb_channels(3);
            cv::split(blob_image, rgb_channels); // ����ͼƬ����ͨ��
            for (auto i = 0; i < rgb_channels.size(); i++) {
                //��ͨ�����˶�ÿһ��ͨ�����ݽ��й�һ������
                rgb_channels[i].convertTo(rgb_channels[i], CV_32FC1, 1.0 / std_values[i], (0.0 - mean_values[i]) / std_values[i]);
            }
            cv::merge(rgb_channels, blob_image); // �ϲ�ͼƬ����ͨ��
        }
        else if (type == 1) {
            // ������ͼƬ����tensor����Ҫ���������
            cv::resize(blob_image, blob_image, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
            // ͼ�����ݹ�һ��
            std::vector<float> std_values{ 255.0,  255.0,  255.0 };
            std::vector<cv::Mat> rgb_channels(3);
            cv::split(blob_image, rgb_channels); // ����ͼƬ����ͨ��
            for (auto i = 0; i < rgb_channels.size(); i++) {
                //��ͨ�����˶�ÿһ��ͨ�����ݽ��й�һ������
                rgb_channels[i].convertTo(rgb_channels[i], CV_32FC1, 1.0 / std_values[i]);
            }
            cv::merge(rgb_channels, blob_image); // �ϲ�ͼƬ����ͨ��
        }
        else if (type == 2) {
            // ��ȡ����任��Ϣ
            cv::Point center(blob_image.cols / 2, blob_image.rows / 2); // �任����
            cv::Size input_size(blob_image.cols, blob_image.rows); // ����ߴ�
            int rot = 0; // �Ƕ�
            cv::Size output_size(width, height); // ����ߴ�

            // ��ȡ����任����
            cv::Mat warp_mat(2, 3, CV_32FC1);
            warp_mat = get_affine_transform(center, input_size, rot, output_size);
            // ����仯
            cv::warpAffine(blob_image, blob_image, warp_mat, output_size, cv::INTER_LINEAR);
            // ͼ�����ݹ�һ��
            std::vector<float> mean_values{ 0.485 * 255, 0.456 * 255, 0.406 * 255 };
            std::vector<float> std_values{ 0.229 * 255, 0.224 * 255, 0.225 * 255 };
            std::vector<cv::Mat> rgb_channels(3);
            cv::split(blob_image, rgb_channels); // ����ͼƬ����ͨ��
            for (auto i = 0; i < rgb_channels.size(); i++) {
                //��ͨ�����˶�ÿһ��ͨ�����ݽ��й�һ������
                rgb_channels[i].convertTo(rgb_channels[i], CV_32FC1, 1.0 / std_values[i], (0.0 - mean_values[i]) / std_values[i]);
            }
            cv::merge(rgb_channels, blob_image); // �ϲ�ͼƬ����ͨ��
        }

        // ԭ��ͼƬ����Ϊ H��W��C ��ʽ������Ҫ���Ϊ C��H��W ��ʽ
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


// @brief ��������任����
// @param center ���ĵ�
// @param input_size ����ߴ�
// @param rot �Ƕ�
// @param output_size ����ߴ�
// @param shift 
// @rrturn �任����
cv::Mat get_affine_transform(cv::Point center, cv::Size input_size, int rot, cv::Size output_size,
    cv::Point2f shift) {

    // ����ߴ���
    int src_w = input_size.width;

    // ����ߴ�
    int dst_w = output_size.width;
    int dst_h = output_size.height;

    // ��ת�Ƕ�
    float rot_rad = 3.1715926f * rot / 180.0;
    int pt = (int)src_w * -0.5;
    float sn = std::sin(rot_rad);
    float cs = std::cos(rot_rad);

    cv::Point2f src_dir(-1.0 * pt * sn, pt * cs);
    cv::Point2f dst_dir(0.0, dst_w * -0.5);
    // ����������
    cv::Point2f src[3];
    src[0] = cv::Point2f(center.x + input_size.width * shift.x, center.y + input_size.height * shift.y);
    src[1] = cv::Point2f(center.x + src_dir.x + input_size.width * shift.x, center.y + src_dir.y + input_size.height * shift.y);
    cv::Point2f direction = src[0] - src[1];
    src[2] = cv::Point2f(src[1].x - direction.y, src[1].y - direction.x);
    // ���������
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


// @brief ��ʼ��PaddleInfer�������
// @return ����PaddleInfer�ṹ��ָ��
extern "C" __declspec(dllexport) void* __stdcall paddle_infer_init() {
    PaddleInfer* paddle_infer = new PaddleInfer(); // ��ʼ��
    return (void*)paddle_infer;
}

// @brief ���ر���Paddleģ��
// @param paddle_infer_ptr PaddleInfer�ṹ��ָ��
// @param model_path_wchar ����ṹ���ļ�·��
// @param params_path_wchar ģ�Ͳ������ļ�·��
// @return ����PaddleInfer�ṹ��ָ��
extern "C" __declspec(dllexport) void* __stdcall set_model(void* paddle_infer_ptr, 
    const wchar_t* model_path_wchar, const wchar_t* params_path_wchar) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    //��ȡ�ӿ��������
    std::string model_path = wchar_to_string(model_path_wchar);// ����ṹ���ļ�·��
    std::string params_path = wchar_to_string(params_path_wchar);// ģ�Ͳ������ļ�·��
    if (model_path == " ") {
        paddle_infer->config.SetModel(model_path); // ֻ����ģ�Ͳ������ļ�·��
    }
    else {
        paddle_infer->config.SetModel(model_path, params_path);
    }
    return (void*)paddle_infer;
}

// @brief ����ģ������ʽΪCPU����
// @param paddle_infer_ptr PaddleInfer�ṹ��ָ��
// @param cpu_math_library_num_threads CPUBlas���߳���
// @return ����PaddleInfer�ṹ��ָ��
extern "C" __declspec(dllexport) void* __stdcall predictor_device_cpu(void* paddle_infer_ptr,
    int cpu_math_library_num_threads) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    paddle_infer->config.DisableGpu(); // ����GPU��ʹ��CPU
    paddle_infer->config.SetCpuMathLibraryNumThreads(cpu_math_library_num_threads); // ����CPUBlas���߳���
    // ��ȡ���ô���Ԥ����
    paddle_infer->predictor = paddle_infer::CreatePredictor(paddle_infer->config);
    return (void*)paddle_infer;
}

// @brief ��������ʽΪGPU����
// @param paddle_infer_ptr PaddleInfer�ṹ��ָ��
// @param memory_init_size ��ʼ��ʹ���Դ��С����λMB
// @param device_id �Կ���ţ�Ĭ��Ϊ0
// @param workspace_size �����Դ��С
// @return ����PaddleInfer�ṹ��ָ��
extern "C" __declspec(dllexport) void* __stdcall predictor_device_gpu(void* paddle_infer_ptr,
    unsigned long long memory_init_size, int device_id, int workspace_size) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    paddle_infer->config.EnableUseGpu(memory_init_size, device_id); // ����GPU��ʹ��GPU����
    paddle_infer->config.SwitchIrOptim(true); // ִ��IRͼ�Ż�
    paddle_infer->config.EnableMemoryOptim(); // ���ڴ��Ż�
    paddle_infer->config.EnableTensorRtEngine(static_cast<int64_t>(1 << workspace_size), 1, 10,
        paddle::AnalysisConfig::Precision::kFloat32, false, false); // ��TensorRT����
    // ��ȡ���ô���Ԥ����
    paddle_infer->predictor = paddle_infer::CreatePredictor(paddle_infer->config);
    return (void*)paddle_infer;
}


// @brief ��������ʽΪONNX runtime
// @param paddle_infer_ptr PaddleInfer�ṹ��ָ��
// @param cpu_math_library_num_threads ONNXRuntime���Ӽ����߳���
// @return ����PaddleInfer�ṹ��ָ��
extern "C" __declspec(dllexport) void* __stdcall predictor_device_ONNX_runtime(void* paddle_infer_ptr,
    int cpu_math_library_num_threads) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    paddle_infer->config.EnableONNXRuntime(); // ���� ONNXRuntime
    paddle_infer->config.EnableORTOptimization();  // ����ONNXRuntime�Ż�
    paddle_infer->config.SetCpuMathLibraryNumThreads(10); // ���� ONNXRuntime ���Ӽ����߳���
    paddle_infer->config.DisableONNXRuntime(); // ���� ONNXRuntime ����Ԥ��
    // ��ȡ���ô���Ԥ����
    paddle_infer->predictor = paddle_infer::CreatePredictor(paddle_infer->config);
    return (void*)paddle_infer;
}

// @brief ��������ʽΪoneDNN
// @param paddle_infer_ptr PaddleInfer�ṹ��ָ��
// @param capacity oneDNN��cache����
// @return ����PaddleInfer�ṹ��ָ��
extern "C" __declspec(dllexport) void* __stdcall predictor_device_oneDNN(void* paddle_infer_ptr,
    int capacity) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    paddle_infer->config.EnableMKLDNN(); // ���� oneDNN ����Ԥ��
    // ���� oneDNN �� cache ����
    // ����̬shape����ʱ���ܻ���n����������shape��Ӧ��oneDNN���ã�����shape�任ʱ�����������ô����Ŀ���
    paddle_infer->config.SetMkldnnCacheCapacity(capacity);
    // ��ȡ���ô���Ԥ����
    paddle_infer->predictor = paddle_infer::CreatePredictor(paddle_infer->config);
    return (void*)paddle_infer;
}

// @brief ��ȡ����ڵ�����
// @param paddle_infer_ptr PaddleInfer�ṹ��ָ��
// @param length �ַ�������
// @return ����ڵ��ַ���
extern "C" __declspec(dllexport) wchar_t* __stdcall get_input_names(void* paddle_infer_ptr,
    int* length) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    // ��ȡ�ڵ�����
    std::vector<std::string> input_names = paddle_infer->predictor->GetInputNames();
    // ���ڵ�תΪ���Դ��ݵĸ�ʽ
    std::string out_str;
    for (int i = 0; i < input_names.size(); i++) {
        out_str += input_names[i];
        length[i] = input_names[i].length();
    }
    return string_to_wchar(out_str);
}


// @brief ����������״
// @param paddle_infer_ptr PaddleInfer�ṹ��ָ��
// @param input_name_wchar ����ڵ�����
// @param input_shape ������״
// @return ����PaddleInfer�ṹ��ָ��
extern "C" __declspec(dllexport) void* __stdcall set_input_shape(void* paddle_infer_ptr,
    wchar_t* input_name_wchar, int* input_shape, int length) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    std::string input_name = wchar_to_string(input_name_wchar);
    std::vector<int> shape(input_shape, input_shape + length);
    std::unique_ptr<paddle_infer::Tensor> input_tensor = paddle_infer->predictor->GetInputHandle(input_name);
    input_tensor->Reshape(shape);
    return (void*)paddle_infer;
}

// @brief ����ͼƬ����
// @param paddle_infer_ptr PaddleInfer�ṹ��ָ��
// @param input_name_wchar ����ڵ�����
// @param image_data ͼƬ����
// @param image_size ͼƬ���ݳ���
// @param type ���ݴ���ʽ
// @return ����PaddleInfer�ṹ��ָ��
extern "C" __declspec(dllexport) void* __stdcall load_input_image_data(void* paddle_infer_ptr,
    wchar_t* input_name_wchar, uchar* image_data, size_t image_size, int type) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    std::string input_name = wchar_to_string(input_name_wchar);
    cv::Mat input_image = data_to_mat(image_data, image_size); // ��������ͼƬ
    // ��ȡ����Tensor
    std::unique_ptr<paddle_infer::Tensor> input_tensor = paddle_infer->predictor->GetInputHandle(input_name);
    std::vector<int> input_shape = input_tensor->shape(); // ��ȡ����Tensor��״
    // ��ȡ��������
    std::vector<cv::Mat> input_images;
    input_images.push_back(input_image);
    std::vector<float> input_datas = input_data_process(input_images, input_shape, type); // ������������
    // ��������
    input_tensor->CopyFromCpu(input_datas.data());
    return (void*)paddle_infer;
}

// @brief ������ͨ����
// @param paddle_infer_ptr PaddleInfer�ṹ��ָ��
// @param input_name_wchar ����ڵ�����
// @param data ��������
// @return ����PaddleInfer�ṹ��ָ��
extern "C" __declspec(dllexport) void* __stdcall load_input_data(void* paddle_infer_ptr,
    wchar_t* input_name_wchar, float* data) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    std::string input_name = wchar_to_string(input_name_wchar);
    // ��ȡ����Tensor
    std::unique_ptr<paddle_infer::Tensor> input_tensor = paddle_infer->predictor->GetInputHandle(input_name);
    std::vector<int> input_shape = input_tensor->shape(); // ��ȡ����Tensor��״
    // ��ȡ���볤��
    int data_length = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int>());
    // ������������
    std::vector<float> input_datas(data_length,1);
    for (int i = 0; i < data_length; i++) {
        input_datas.push_back(data[i]);
    }
    input_tensor->CopyFromCpu(input_datas.data()); // ������������
    return (void*)paddle_infer;
}


// @brief ģ������
// @param paddle_infer_ptr PaddleInfer�ṹ��ָ��
// @return ����PaddleInfer�ṹ��ָ��
extern "C" __declspec(dllexport) void* __stdcall infer(void* paddle_infer_ptr) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    paddle_infer->predictor->Run();
    return (void*)paddle_infer;
}


// @brief ��ȡ����ڵ�����
// @param paddle_infer_ptr PaddleInfer�ṹ��ָ��
// @param length �ַ�������
// @return ����ڵ��ַ���
extern "C" __declspec(dllexport) wchar_t* __stdcall get_output_names(void* paddle_infer_ptr,
    int* length) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    // ��ȡ�ڵ�����
    std::vector<std::string> input_names = paddle_infer->predictor->GetOutputNames();
    // ���ڵ�תΪ���Դ��ݵĸ�ʽ
    std::string out_str;
    for (int i = 0; i < input_names.size(); i++) {
        out_str += input_names[i];
        length[i] = input_names[i].length();
    }
    return string_to_wchar(out_str);
}

// @brief ��ȡָ���ڵ����״
// @param paddle_infer_ptr PaddleInfer�ṹ��ָ��
// @param node_name_wchar �ڵ���
// @param shape ��״
// @param dimension ά��
// @return ������
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

// @brief ��ȡģ�ͽ�����-F32
// @param paddle_infer_ptr PaddleInfer�ṹ��ָ��
// @param output_name_wchar ����ڵ���
// @return ������
extern "C" __declspec(dllexport) void __stdcall read_result_data_F32(void* paddle_infer_ptr,
    wchar_t * output_name_wchar, float* infer_result) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    std::string output_name = wchar_to_string(output_name_wchar);
    // ��ȡ����ڵ���
    std::unique_ptr<paddle_infer::Tensor> output_tensor = paddle_infer->predictor->GetOutputHandle(output_name);
    std::vector<int> output_shape = output_tensor->shape();
    int data_length = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    std::vector<float> result(data_length);
    output_tensor->CopyToCpu(result.data()); // ��ȡ���
    for (int i = 0; i < data_length; i++) {
        *infer_result = result[i];
        infer_result++;
    }
}

// @brief ��ȡģ�ͽ�����-I32
// @param paddle_infer_ptr PaddleInfer�ṹ��ָ��
// @param output_name_wchar ����ڵ���
// @return ������
extern "C" __declspec(dllexport) void __stdcall read_result_data_I32(void* paddle_infer_ptr,
    wchar_t* output_name_wchar, int* infer_result) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    std::string output_name = wchar_to_string(output_name_wchar);
    // ��ȡ����ڵ���
    std::unique_ptr<paddle_infer::Tensor> output_tensor = paddle_infer->predictor->GetOutputHandle(output_name);
    std::vector<int> output_shape = output_tensor->shape();
    int data_length = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    std::vector<int> result(data_length);
    output_tensor->CopyToCpu(result.data()); // ��ȡ���
    for (int i = 0; i < data_length; i++) {
        *infer_result = result[i];
        infer_result++;
    }
}
// @brief ��ȡģ�ͽ�����-I64
// @param paddle_infer_ptr PaddleInfer�ṹ��ָ��
// @param output_name_wchar ����ڵ���
// @return ������
extern "C" __declspec(dllexport) void __stdcall read_result_data_I64(void* paddle_infer_ptr,
    wchar_t* output_name_wchar, long long* infer_result) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    std::string output_name = wchar_to_string(output_name_wchar);
    // ��ȡ����ڵ���
    std::unique_ptr<paddle_infer::Tensor> output_tensor = paddle_infer->predictor->GetOutputHandle(output_name);
    std::vector<int> output_shape = output_tensor->shape();
    int data_length = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    std::vector<long long> result(data_length);
    output_tensor->CopyToCpu(result.data()); // ��ȡ���
    for (int i = 0; i < data_length; i++) {
        *infer_result = result[i];
        infer_result++;
    }
}



// @brief �����ڴ�
// @param paddle_infer_ptr PaddleInfer�ṹ��ָ��
extern "C" __declspec(dllexport) void __stdcall dispose(void* paddle_infer_ptr) {
    PaddleInfer* paddle_infer = (PaddleInfer*)paddle_infer_ptr;
    delete paddle_infer;
}


