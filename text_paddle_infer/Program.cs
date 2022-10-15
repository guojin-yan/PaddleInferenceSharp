using System;
using OpenCvSharp;
using PaddleInferenceSharp;
namespace text_paddle_infer // Note: actual namespace depends on the project name.
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            string model_path = "E:/Text_Model/flowerclas/inference.pdmodel";
            string params_path = "E:/Text_Model/flowerclas/inference.pdiparams";
            string image_path = "E:/Text_dataset/flowers102/jpg/image_00005.jpg";
            IntPtr ptr = NativeMethods.paddle_infer_init();
            ptr = NativeMethods.set_model(ptr, model_path, params_path);
            ptr = NativeMethods.predictor_device_cpu(ptr, 10);
            int[] input_size = new int[4] { 1, 3, 224, 224 };
            ptr = NativeMethods.set_input_shape(ptr, "x", ref input_size[0], 4);
            Mat image = Cv2.ImRead(image_path);
            byte[] input_image_data = image.ImEncode(".bmp");
            // 数据长度
            ulong input_image_length = Convert.ToUInt64(input_image_data.Length);
            ptr = NativeMethods.load_input_image_data(ptr, "x", ref input_image_data[0], input_image_length, 0);
            ptr = NativeMethods.infer(ptr);
            int[] leng = new int[4];
            string input_name = NativeMethods.get_output_names(ptr, ref leng[0]);
            Console.WriteLine(input_name);
            Console.WriteLine(leng[0]);
            float[] output = new float[102];
            NativeMethods.read_result_data_F32(ptr, input_name, ref output[0]);
            for (int i = 0; i < 102; i++) 
            {
                Console.WriteLine(output[i]);
            }
        }
    }
}