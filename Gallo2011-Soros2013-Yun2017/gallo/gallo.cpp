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

#include <iostream>
#include "gallo.h"

using namespace iy;

cv::Rect Gallo::process(cv::Mat &gray_src, int WinSz/*=20*/)
{
    cv::setNumThreads(0);
    cv::Rect result(0,0,0,0);
    
    try{
       // gradient map
       cv::Mat hGrad = calc_gradient(gray_src);
       
       // integral map
       cv::Mat iMap = calc_integral_image(hGrad);
       
       // find max point with box filter
       cv::Mat sMap(hGrad.size(), CV_8UC1); 
       cv::Point cp = find_max_point_with_smooth(iMap, sMap, WinSz);
    
       // global binzrization
       cv::Mat bMap(hGrad.size(), CV_8UC1);
       cv::threshold(sMap, bMap, 50, 255, cv::THRESH_OTSU);
       
       // box detection       
       result = box_detection(bMap, cp);
    }
    catch(cv::Exception &e)
    {
        std::cerr << "cv::Exception: " << std::endl;
        std::cerr << e.what() << std::endl;
    }
    
    // return result
    return result;
}

cv::Mat Gallo::calc_gradient(cv::Mat &src)
{
    assert(src.channels() == 1);
    
    const cv::Size imSz = src.size();    
    cv::Mat result(imSz, CV_8UC1);
    
    for(int h = 1; h < imSz.height - 1; h++)
    {
        for(int w = 1; w < imSz.width - 1; w++)
        {
            int dx = src.at<uchar>(h-1,w-1) + 2*src.at<uchar>(h,w-1) + src.at<uchar>(h+1,w-1) - 
                     src.at<uchar>(h-1,w+1) - 2*src.at<uchar>(h,w+1) - src.at<uchar>(h+1,w+1);
            
            result.at<uchar>(h,w) = std::abs(dx); 
        }
    }    
    
    return result;   
}

cv::Mat Gallo::calc_integral_image(cv::Mat &src)
{
    assert(src.channels() == 1);
    
    const cv::Size imSz = src.size();    
    cv::Mat result(imSz, CV_32FC1);
    
    // -1st cols
    result.at<float>(0,0) = src.at<uchar>(0,0);
    for(int w = 1; w < imSz.width; w++)
        result.at<float>(0, w) = (float)src.at<uchar>(0, w) + result.at<float>(0, w-1);
    
    // -other cols
    for(int h = 1; h < imSz.height; h++)
    {
        float sum = 0.0f;
        for(int w = 0; w < imSz.width; w++)
        {
            sum += src.at<uchar>(h, w);
            result.at<float>(h, w) = result.at<float>(h-1, w) + sum;
        }
    }
    
    return result;    
}

cv::Point Gallo::find_max_point_with_smooth(cv::Mat &src, cv::Mat &smooth_map, int WinSz)
{
    const cv::Size imSz = src.size(); 
    cv::Point max_pt(0, 0);
    
    const int nSize = WinSz * WinSz;
    const int cSize = (WinSz / 2) + 1;
    const int max_height =  imSz.height - cSize;
    const int max_width = imSz.width - cSize;
    float mean_max = 0.0f;
    
    for(int h = 0; h < imSz.height; h++)
    {
        int temp_top = h - cSize;
        int ntop = (temp_top > max_height) ? max_height : temp_top;
        int temp_bottom = h + cSize;
        int nbottom = (temp_bottom >= imSz.height-1) ? imSz.height-1 : temp_bottom; 
        
        for(int w = 0; w < imSz.width; w++)
        {
            int temp_left = w - cSize;
            int nleft = (temp_left > max_width) ? max_width : temp_left;
            int temp_right = w + cSize;
            int nright = (temp_right >= imSz.width-1) ? imSz.width-1 : temp_right; 
            
            // local mean
            float n1 = (nleft > 0 && ntop > 0) ? src.at<float>(ntop, nleft-1): 0;
            float n2 = (nleft > 0) ? src.at<float>(nbottom, nleft-1): 0;
            float n3 = (ntop > 0) ? src.at<float>(ntop, nright): 0;
            
            float sum = src.at<float>(nbottom, nright) - n3 - n2 + n1;
            float mean = sum / nSize;
            
            if(mean > mean_max){
                mean_max = mean;
                max_pt.x = w;
                max_pt.y = h;
            }
            
            smooth_map.at<uchar>(h,w) = (mean > 255) ? 255 : mean;
        }
    }
    
    return max_pt;
}

cv::Rect Gallo::box_detection(cv::Mat &src, cv::Point cp)
{
    cv::Rect result(0,0,0,0);
    
    const cv::Size imSz = src.size();
    
    // lfet
    for(int w = cp.x; w >= 0; w--)
    {
        if(src.at<uchar>(cp.y, w) < 128)
        {
            result.x = w;
            break;
        }
    }
    
    // right
    for(int w = cp.x; w < imSz.width; w++)
    {
        if(src.at<uchar>(cp.y, w) < 128)
        {
            result.width = w - result.x;
            break;
        }
    }
    
    // up
    for(int h = cp.y; h >= 0; h--)
    {
        if(src.at<uchar>(h, cp.x) < 128)
        {
            result.y = h;
            break;
        }
    }
    
    //down
    for(int h = cp.y; h < imSz.height; h++)
    {
        if(src.at<uchar>(h, cp.x) < 128)
        {
            result.height = h - result.y;
            break;
        }
    }
    
    return result;
}










