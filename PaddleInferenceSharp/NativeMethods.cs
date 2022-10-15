using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace PaddleInferenceSharp
{
    public class NativeMethods
    {
        private const string paddle_infer_path = @"E:\Git_space\PaddleInferenceSharp\paddle_infrer_api\dll\PaddleInferAPI.dll";
        [DllImport(paddle_infer_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr paddle_infer_init();
        [DllImport(paddle_infer_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr set_model(IntPtr paddle_infer, string model_path, string params_path);
        [DllImport(paddle_infer_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr predictor_device_cpu(IntPtr paddle_infer, int cpu_math_library_num_threads);
        [DllImport(paddle_infer_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr predictor_device_gpu(IntPtr paddle_infer, ulong memory_init_size, int device_id, int workspace_size);
        [DllImport(paddle_infer_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr predictor_device_ONNX_runtime(IntPtr paddle_infer, int cpu_math_library_num_threads);
        [DllImport(paddle_infer_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr predictor_device_oneDNN(IntPtr paddle_infer, int capacity);
        [DllImport(paddle_infer_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public static extern string get_input_names(IntPtr paddle_infer, ref int length);
        [DllImport(paddle_infer_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr set_input_shape(IntPtr paddle_infer, string input_name, ref int input_shape, int length);
        [DllImport(paddle_infer_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr load_input_image_data(IntPtr paddle_infer, string input_name, ref byte image_data, ulong image_size, int type);
        [DllImport(paddle_infer_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr load_input_data(IntPtr paddle_infer, string input_name, ref float data);
        [DllImport(paddle_infer_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr infer(IntPtr paddle_infer);
        [DllImport(paddle_infer_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public static extern string get_output_names(IntPtr paddle_infer, ref int length);
        [DllImport(paddle_infer_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public static extern void read_result_data_F32(IntPtr paddle_infer, string output_name, ref float result);
        [DllImport(paddle_infer_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public static extern void read_result_data_I32(IntPtr paddle_infer, string output_name, ref int result);
        [DllImport(paddle_infer_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public static extern void read_result_data_I64(IntPtr paddle_infer, string output_name, ref long result);
        [DllImport(paddle_infer_path, CharSet = CharSet.Unicode, CallingConvention = CallingConvention.Cdecl)]
        public static extern void dispose(IntPtr paddle_infer);
           

    }
}
