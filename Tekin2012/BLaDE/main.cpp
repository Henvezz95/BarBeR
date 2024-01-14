#include <string>
#include <vector>
#include "ski/BLaDE/Barcode.h"
#include "ski/BLaDE/Symbology.h"
#include "ski/types.h"
#include <memory>
#include "Locator.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "wrapper.h"

int main(){
    //std::string file_name = "/media/sf_evezzali/OneDrive - Unimore/PhD Resources/Datasets/Artelab 1D barcodes/Dataset1/05102009101.jpg";
    cv::Mat img_color;
    cv::Mat img_color2;
    cv::Mat img_color3;
    std::string file_name;
    for(int i=0 ; i<10  ; i++){
        file_name = "/home/dl.net/evezzali/Gitlab/Barcode-Localization-Benchmark/dataset/images/EAN13_09_0003.jpg";
        img_color = cv::imread(file_name.c_str());
        cv::resize(img_color, img_color, cv::Size(640, 480));
        int result[10000];
        int num_results = 0;
        locateBarcodes(result, &num_results, img_color.data, img_color.rows, img_color.cols);
        std::cout << num_results << std::endl;
        file_name = "/home/dl.net/evezzali/Gitlab/Barcode-Localization-Benchmark/dataset/images/EAN13_09_0001.jpg";
        img_color2 = cv::imread(file_name.c_str());
        cv::resize(img_color2, img_color2, cv::Size(320, 240));
        locateBarcodes(result, &num_results, img_color2.data, img_color2.rows, img_color2.cols);
        std::cout << num_results << std::endl;
        file_name = "/home/dl.net/evezzali/Gitlab/Barcode-Localization-Benchmark/dataset/images/EAN13_09_0003.jpg";
        img_color3 = cv::imread(file_name.c_str());
        cv::resize(img_color3, img_color3, cv::Size(320, 240));
        locateBarcodes(result, &num_results, img_color3.data, img_color3.rows, img_color3.cols);
        std::cout << num_results << std::endl;

    }



}