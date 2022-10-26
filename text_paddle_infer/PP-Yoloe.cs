using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using PaddleInferenceSharp;
using OpenCvSharp;

namespace text_paddle_infer
{
    internal class PP_Yoloe
    {
        public static void paddle_deploy_ppyoloe() 
        {
            //----------------------1.模型相关信息----------------------//
            string model_path = @"E:\Text_Model\ppyoloe_plus_crn_l_80e_coco\model.pdmodel";
            string params_path = @"E:\Text_Model\ppyoloe_plus_crn_l_80e_coco\model.pdiparams";
            string image_path = @"E:\Text_dataset\YOLOv5\0003.jpg";
            DateTime begin, end;
            TimeSpan t0, t1, t2, t3;
            //----------------------2. 创建并配置预测器------------------//
            begin = DateTime.Now;
            PaddleInfer predictor = new PaddleInfer(model_path, params_path);
            // 设置设备类型
            //predictor.set_divice(Divice.CPU, 10); // CPU
            //predictor.set_divice(Divice.GPU, 0, 500, 30); // GPU
            predictor.set_divice(Divice.ONNX_runtime, 10); // ONNX_runtime
            //predictor.set_divice(Divice.oneDNN, 1); // oneDNN
            // 获取输入节点
            List<string> input_name = predictor.get_input_names();
            for (int i = 0; i < input_name.Count; i++)
            {
                Console.WriteLine("模型输入 {0}  {1}", i, input_name[i]);
            }
            // 设置输入形状
            int[] input_size1 = new int[4] { 1, 3, 640, 640 };
            predictor.set_input_shape(input_size1, input_name[0]);
            int[] input_size2 = new int[2] { 1,2};
            predictor.set_input_shape(input_size2, input_name[1]);
            end = DateTime.Now;
            t0 = end - begin;
            //----------------------3. 加载推理数据------------------//
            begin = DateTime.Now;
            Mat image = Cv2.ImRead(image_path);
            // 将图片放在矩形背景下
            int max_image_length = image.Cols > image.Rows ? image.Cols : image.Rows;
            Mat max_image = Mat.Zeros(new Size(max_image_length, max_image_length), MatType.CV_8UC3);
            Rect roi = new Rect(0, 0, image.Cols, image.Rows);
            image.CopyTo(new Mat(max_image, roi));
            byte[] input_image_data = max_image.ImEncode(".bmp");
            // 数据长度
            ulong input_image_length = Convert.ToUInt64(input_image_data.Length);
            predictor.load_input_data(input_name[0], input_image_data, input_image_length, 2);
            float scale_factor = 640.0f/ max_image_length;
            float[] input_scale = new float[] { scale_factor, scale_factor };
            predictor.load_input_data(input_name[1], input_scale);
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
            List<int> output_shape = predictor.get_shape(output_name[0]);
            Console.WriteLine("output_shape：{0} × {1}", output_shape[0], output_shape[1]);
            int[] output_num = predictor.read_infer_result<int>(output_name[1], 1);
            Console.WriteLine(output_num[0]);
            float[] output = predictor.read_infer_result<float>(output_name[0], output_num[0] * 6);
            List<Rect> boxes = new List<Rect>();
            List<int> classes = new List<int>();
            List<float> scores = new List<float>();
            for (int i = 0; i < output_num[0]; i++)
            {
                if (output[6 * i + 1] > 0.4)
                { 
                    scores.Add(output[6 * i + 1]);
                    classes.Add((int)output[6 * i]);
                    Rect rect = new Rect((int)output[6 * i + 2], (int)output[6 * i + 3],
                        (int)(output[6 * i + 4] - output[6 * i + 2]), (int)(output[6 * i + 5] - output[6 * i + 3]));
                    boxes.Add(rect);
                }
                
            }
            end = DateTime.Now;
            t3 = end - begin;
            Console.WriteLine("模型加载时间：{0}", t0.TotalMilliseconds);
            Console.WriteLine("推理数据加载时间：{0}", t1.TotalMilliseconds);
            Console.WriteLine("模型推理时间：{0}", t2.TotalMilliseconds);
            Console.WriteLine("结果处理时间：{0}", t3.TotalMilliseconds);

            for (int i = 0; i < classes.Count; i++) 
            {
                Cv2.Rectangle(image, boxes[i], new Scalar(0, 0, 255), 1, LineTypes.Link8);
                Cv2.PutText(image, scores[i].ToString(), new Point(boxes[i].X, boxes[i].Y - 5),
                    HersheyFonts.HersheySimplex, 0.5, new Scalar(0, 255, 0));
            }
            Cv2.ImShow("result", image);
            Cv2.WaitKey(0);
        }
    }
}
