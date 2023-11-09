/*
 *  Copyright 2014-2017 Inyong Yun (Sungkyunkwan University)
 *
 *  Created on: 2017. 11. 24. (Rev. 2.0)
 *      Author: Inyong Yun 
 *        type: c/c++
 * 
 *   Reference paper: 
 *   Gallo, O., & Manduchi, R. (2011). "Reading 1D barcodes with mobile phones using deformable templates." 
 *   IEEE transactions on pattern analysis and machine intelligence, 33(9), 1834-1843.
 *   
 *   etc: not include barcode recognition process.
 *        only localization method! *    
 */
#ifndef GALLO
#define GALLO
#include <opencv2/opencv.hpp>

namespace iy{
    class Gallo {
    private:        
        cv::Mat calc_gradient(cv::Mat &src);
        cv::Mat calc_integral_image(cv::Mat &src);
        cv::Point find_max_point_with_smooth(cv::Mat &src, cv::Mat &smooth_map, int WinSz);
        cv::Rect box_detection(cv::Mat &src, cv::Point cp);
    public:
        Gallo(){}   
        ~Gallo(){}  
        
        cv::Rect process(cv::Mat &gray_src, int WinSz = 20);
    };
};
#endif