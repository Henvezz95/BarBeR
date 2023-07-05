#include <opencv2/imgcodecs.hpp> 
#include <opencv2/core/core.hpp>
#include "barcode_localization.h"
#include <chrono>

int main() {
    cv::Mat img = cv::imread("../05102009081.jpg", cv::IMREAD_COLOR);
    std::cout<<"0"<<std::endl;
    int result[10000];
    uint8_t *dataindex=(uint8_t*)img.datastart;
    int minLineLength = 200;
    int support_candidates_threshold = 10;
    int delta = 5;
    int maxLengthToLineLengthRatio = 6;
    int minLengthToLineLengthRatio = 1;

    
    locateBarcode(result, dataindex, img.rows, img.cols, minLineLength, support_candidates_threshold, delta, maxLengthToLineLengthRatio, minLengthToLineLengthRatio, 3,3);
    std::cout <<result[0]<<std::endl;

}


