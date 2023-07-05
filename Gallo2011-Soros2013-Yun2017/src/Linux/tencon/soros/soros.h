/*
 *  Copyright 2014-2017 Inyong Yun (Sungkyunkwan University)
 *
 *  Created on: 2017. 11. 24. (Rev. 2.0)
 *      Author: Inyong Yun 
 *        type: c/c++
 * 
 *   Reference paper: 
 *   Sörös, Gábor, and Christian Flörkemeier. "Blur-resistant joint 1D and 2D barcode localization for smartphones." 
 *   Proceedings of the 12th International Conference on Mobile and Ubiquitous Multimedia. ACM, 2013.
 *
 *   etc: not include barcode recognition process.
 *        only localization method! *    
 */
#ifndef SOROS
#define SOROS
#include <opencv2/opencv.hpp>

namespace iy{
    class Soros {
    private:        
        cv::Mat SaliencyMapbyAndoMatrix(cv::Mat &src, bool is1D = true);        
        cv::Mat calc_integral_image(cv::Mat &src);
        cv::Point find_max_point_with_smooth(cv::Mat &src, cv::Mat &smooth_map, int WinSz = 20);
        cv::Rect box_detection(cv::Mat &src, cv::Point cp);
    public:
        Soros() {}  
        ~Soros() {} 
        
        cv::Rect process(cv::Mat &gray_src, bool is1D = true, int WinSz = 20);
    };
}
#endif