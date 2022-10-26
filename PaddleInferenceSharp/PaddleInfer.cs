using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PaddleInferenceSharp
{
    /// <summary>
    /// PaddlePaddle Inference 模型推理类
    /// </summary>
    public class PaddleInfer
    {
        // 推理核心地址
        private IntPtr paddle_infer;

        /// <summary>
        /// 构造函数
        /// </summary>
        /// <param name="model_path">模型地址</param>
        /// <param name="params_path">模型参数地址</param>
        public PaddleInfer(string model_path, string params_path=" ") 
        {
            paddle_infer = NativeMethods.paddle_infer_init();
            paddle_infer = NativeMethods.set_model(paddle_infer, model_path, params_path);
        }
        /// <summary>
        /// 设置推理设备
        /// 0.CPU，1.GPU，2.ONNX runtime，3.oneDNN
        /// </summary>
        /// <param name="divice">设备选择</param>
        /// <param name="num">对于CPU、ONNX runtime代表线程数，默认为10；对于GPU代表显卡编号，默认为0；对于oneDNN代表cache数量，默认为1</param>
        /// <param name="memory_init_size">显存分配空间(尽在使用GPU时作用)</param>
        /// <param name="workspace_size">显存工作空间(尽在使用GPU时作用)</param>
        public void set_divice(Divice divice, int num = 0, ulong memory_init_size = 500, int workspace_size = 30)
        {
            if (divice == Divice.CPU)
            {
                if (num == 0)
                {
                    num = 10;
                }
                paddle_infer = NativeMethods.predictor_device_cpu(paddle_infer, num);
            }
            else if (divice == Divice.ONNX_runtime)
            {
                if (num == 0)
                {
                    num = 10;
                }
                paddle_infer = NativeMethods.predictor_device_ONNX_runtime(paddle_infer, num);
            }
            else if (divice == Divice.oneDNN)
            {
                if (num == 0)
                {
                    num = 1;
                }
                paddle_infer = NativeMethods.predictor_device_oneDNN(paddle_infer, num);
            }
            else if (divice == Divice.GPU) 
            {
                paddle_infer = NativeMethods.predictor_device_gpu(paddle_infer, memory_init_size, num, workspace_size);
            }
        }
        /// <summary>
        /// 获取输入节点名字
        /// </summary>
        /// <returns>输入节点列表</returns>
        public List<string> get_input_names()
        {
            int[] length = new int[5];
            string input = NativeMethods.get_input_names(paddle_infer,ref length[0]);
            List<string> input_name = new List<string>();
            int l = 0;
            for (int i = 0; i < length.Length; i++) 
            {
                string str = input.Substring(l, length[i]);
                input_name.Add(str);
                l += length[i];
                if (l >= input.Length) 
                {
                    break;
                }
            }
            return input_name;
        }
        /// <summary>
        /// 设置输入节点形状
        /// </summary>
        /// <param name="input_shape">形状数组</param>
        /// <param name="input_name">节点名称</param>
        public void set_input_shape(int[] input_shape, string input_name) 
        {
            paddle_infer = NativeMethods.set_input_shape(paddle_infer, input_name, ref input_shape[0], input_shape.Length);
        }

        /// <summary>
        /// 加载普通数据
        /// </summary>
        /// <param name="input_name">输入节点名称</param>
        /// <param name="input_data">输入数据</param>
        public void load_input_data(string input_name, float[] input_data)
        {
            paddle_infer = NativeMethods.load_input_data(paddle_infer, input_name, ref input_data[0]);
        }
        /// <summary>
        ///  加载图片数据
        /// </summary>
        /// <param name="input_name">输入节点名称</param>
        /// <param name="image_data">图片数据</param>
        /// <param name="image_size">图片长度</param>
        /// <param name="type">数据处理类型</param>
        public void load_input_data(string input_name, byte[] image_data, ulong image_size, int type)
        {
            paddle_infer = NativeMethods.load_input_image_data(paddle_infer, input_name, ref image_data[0], image_size, type);
        }
        /// <summary>
        /// 模型推理
        /// </summary>
        public void infer()
        {
            paddle_infer = NativeMethods.infer(paddle_infer);
        }
        /// <summary>
        /// 获取输出节点名字
        /// </summary>
        /// <returns>输出节点列表</returns>
        public List<string> get_output_names()
        {
            int[] length = new int[5];
            string output = NativeMethods.get_output_names(paddle_infer, ref length[0]);
            List<string> output_name = new List<string>();
            int l = 0;
            for (int i = 0; i < length.Length; i++)
            {
                string str = output.Substring(l, length[i]);
                output_name.Add(str);
                l += length[i];
                if (l >= output.Length)
                {
                    break;
                }
            }
            return output_name;
        }
        /// <summary>
        /// 获取指定节点形状
        /// </summary>
        /// <param name="node_name">节点名称</param>
        /// <returns></returns>
        public List<int> get_shape(string node_name) 
        {
            int[] shape = new int[5];
            int dimension = 0;
            List<int> shape_out = new List<int>();
            NativeMethods.get_node_shape(paddle_infer, node_name, ref shape[0], ref dimension);
            for (int i = 0; i < dimension; i++) 
            {
                shape_out.Add(shape[i]);
            }
            return shape_out;
        }

        /// <summary>
        /// 读取推理结果数据
        /// </summary>
        /// <typeparam name="T">数据类型</typeparam>
        /// <param name="output_name">输出节点名</param>
        /// <param name="data_size">输出数据长度</param>
        /// <returns>推理结果数组</returns>
        public T[] read_infer_result<T>(string output_name, int data_size)
        {
            // 获取设定类型
            string t = typeof(T).ToString();
            // 新建返回值数组
            T[] result = new T[data_size];
            if (t == "System.Int32")
            { // 读取数据类型为整形数据
                int[] inference_result = new int[data_size];
                NativeMethods.read_result_data_I32(paddle_infer, output_name, ref inference_result[0]);
                result = (T[])Convert.ChangeType(inference_result, typeof(T[]));
                return result;
            }
            else if (t == "System.Int64")
            {
                long[] inference_result = new long[data_size];
                NativeMethods.read_result_data_I64(paddle_infer, output_name, ref inference_result[0]);
                result = (T[])Convert.ChangeType(inference_result, typeof(T[]));
                return result;
            }
            else
            { // 读取数据类型为浮点型数据
                float[] inference_result = new float[data_size];
                NativeMethods.read_result_data_F32(paddle_infer, output_name, ref inference_result[0]);
                result = (T[])Convert.ChangeType(inference_result, typeof(T[]));
                return result;
            }
        }
        /// <summary>
        /// 删除内存地址
        /// </summary>
        public void delet()
        {
            NativeMethods.dispose(paddle_infer);
        }

    }


    /// <summary>
    /// 设备名称
    /// </summary>
    public enum Divice { 
        CPU,
        GPU,
        ONNX_runtime,
        oneDNN
    }
}