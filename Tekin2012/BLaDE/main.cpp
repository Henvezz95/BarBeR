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
    std::string file_name = "/media/sf_evezzali/OneDrive - Unimore/PhD Resources/Datasets/Artelab 1D barcodes/Dataset1/05102009101.jpg";
    //std::string file_name = "/media/sf_evezzali/OneDrive - Unimore/PhD Resources/Datasets/Artelab 1D barcodes/Dataset1/05102009081.jpg";
    cv::Mat img_color = cv::imread(file_name.c_str());
    cv::resize(img_color, img_color, cv::Size(640, 480));
    int result[10000];
    int num_results = 0;
    locateBarcodes(result, &num_results, img_color.data, img_color.rows, img_color.cols);
    std::cout << num_results << std::endl;


}