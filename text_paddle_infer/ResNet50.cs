using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using PaddleInferenceSharp;
using OpenCvSharp;

namespace text_paddle_infer
{
    internal class ResNet50
    {
        public static void paddle_deploy_resnet50() 
        {
            //----------------------1.模型相关信息----------------------//
            string model_path = "E:/Text_Model/flowerclas/inference.pdmodel";
            string params_path = "E:/Text_Model/flowerclas/inference.pdiparams";
            string image_path = "E:/Text_dataset/flowers102/jpg/image_00005.jpg";
            DateTime begin, end;
            TimeSpan t0, t1, t2, t3;
            //----------------------2. 创建并配置预测器------------------//
            begin = DateTime.Now;
            PaddleInfer predictor = new PaddleInfer(model_path, params_path);
            // 设置设备类型
            predictor.set_divice(Divice.CPU, 10); // CPU
            //predictor.set_divice(Divice.GPU, 0, 500, 30); // GPU
            //predictor.set_divice(Divice.ONNX_runtime, 10); // ONNX_runtime
            //predictor.set_divice(Divice.oneDNN, 1); // oneDNN
            // 获取输入节点
            List<string> input_name = predictor.get_input_names();
            for (int i = 0; i < input_name.Count; i++)
            {
                Console.WriteLine("模型输入 {0}  {1}", i, input_name[i]);
            }
            // 设置输入形状
            int[] input_size = new int[4] { 1, 3, 224, 224 };
            predictor.set_input_shape(input_size, input_name[0]);
            end = DateTime.Now;
            t0 = end - begin;
            //----------------------3. 加载推理数据------------------//
            begin = DateTime.Now;
            Mat image = Cv2.ImRead(image_path);
            byte[] input_image_data = image.ImEncode(".bmp");
            // 数据长度
            ulong input_image_length = Convert.ToUInt64(input_image_data.Length);
            predictor.load_input_data(input_name[0], input_image_data, input_image_length, 0);
            end = DateTime.Now;
            t1 = end - begin;
            //----------------------4. 模型推理------------------//
            begin = DateTime.Now;
            predictor.infer();
            end = DateTime.Now;
            t2 = end - begin;
            //----------------------5. 模型推理结果------------------//
            begin = DateTime.Now;
            int[] leng = new int[4];
            List<string> output_name = predictor.get_output_names();
            for (int i = 0; i < input_name.Count; i++)
            {
                Console.WriteLine("模型输出 {0}  {1}", i, output_name[i]);
            }
            float[] output = predictor.read_infer_result<float>(output_name[0], 102);
            float max;
            int index = max_indax<float>(output, out max);
            end = DateTime.Now;
            t3 = end - begin;
            Console.WriteLine("最大类别为：{0}，分数：{1}。", index, max);
            Console.WriteLine("模型加载时间：{0}", t0.TotalMilliseconds);
            Console.WriteLine("推理数据加载时间：{0}", t1.TotalMilliseconds);
            Console.WriteLine("模型推理时间：{0}", t2.TotalMilliseconds);
            Console.WriteLine("结果处理时间：{0}", t3.TotalMilliseconds);


        }

        static int max_indax<T>(T[] data, out T max) where T : IComparable<T>
        { 
            int index = 0;
            max = data[0];
            for (int i = 0; i < data.Length; i++) 
            {
                if (data[i].CompareTo(max) > 0)
                {
                    index = i;
                    max = data[i];
                }
            }
            



            return index;
        }
    }
}
