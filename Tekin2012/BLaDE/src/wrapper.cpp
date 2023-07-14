#include "wrapper.h"
#include <string>
#include <vector>
#include "ski/BLaDE/Barcode.h"
#include "ski/BLaDE/Symbology.h"
#include "ski/types.h"
#include <memory>
#include "Locator.h"
#include <opencv2/opencv.hpp>


void locateBarcodes(int* result, int* num_results, unsigned char* img_color, int h, int w){
    
    cv::Mat image_color(h, w, CV_8UC3, img_color);
    cv::Mat image_greyscale;
    cv::cvtColor(image_color, image_greyscale, cv::COLOR_BGR2GRAY);
    TMatrixUInt8 new_img(h, w, image_greyscale.data);
    BarcodeLocator *bl = new BarcodeLocator(new_img);
    BarcodeList barcodeList;
    bl->locate(barcodeList);
    (*num_results) = (int)barcodeList.size();
    BarcodeList::iterator it;
    int i = 0;
    for(it = barcodeList.begin(); it != barcodeList.end(); ++it){
        result[i*4] = it->firstEdge.x;
        result[i*4+1] = it->firstEdge.y;
        result[i*4+2] = it->lastEdge.x;
        result[i*4+3] = it->lastEdge.y;
        i++;
    }
}