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
            //ResNet50.paddle_deploy_resnet50();
            PP_Yoloe.paddle_deploy_ppyoloe();
        }
    }
}