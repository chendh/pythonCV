# 數位影像處理概論
### 使用python 與 Opencv實作

* 課程前置工作
  - OpenCV簡介
  - 使用OpenCV開發應用程式
  - 準備工作
    * C++ (Visual Studio)
    * 安裝git
    * 使用VCPKG
    * .\vcpkg.exe search opencv
    * .\vcpkg.exe install opencv:x64-windows
    * .\vcpkg.exe integrate project
    * 在VS 工具選單中使用NuGet ->套件管理管理器主控台
    ```
    #include "opencv2/imgcodecs.hpp"
    #include "opencv2/highgui.hpp"
    #include "opencv2/imgproc.hpp"
    #include <iostream>
    using namespace cv;
    using namespace std;
    int main(int argc, char** argv)
    {
    //CommandLineParser parser(argc, argv, "{@input | lena.jpg | input image}");
    Mat src = imread("image/cameraman.png", IMREAD_COLOR);
    if (src.empty())
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }
    cvtColor(src, src, COLOR_BGR2GRAY);
    Mat dst;
    equalizeHist(src, dst);
    imshow("Source image", src);
    imshow("Equalized Image", dst);
    waitKey();
    return 0;
    }

    ```

* 點處理
* 鄰域處理
* 幾何轉換
* 使用tkinter新增視窗元件
* 
