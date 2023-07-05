#include "ImageProcessor.hpp"
#include "wrapper.h"

#define foreach BOOST_FOREACH
#define barcodeimage ArtelabDataset::barcode_image

using namespace artelab;
using std::string;
using std::cout;
using std::flush;
using std::endl;
using std::vector;


cv::Size win_size(61, 3);
ImageProcessor *pr;

void initialize(const char* nn_path){
    pr = new ImageProcessor(std::string(nn_path), win_size);
}

void locateBarcode(int* result, double* angle, int* num_results, unsigned char* img_color, int h, int w){
    pr->locate(result, angle, num_results, img_color, h, w);
}