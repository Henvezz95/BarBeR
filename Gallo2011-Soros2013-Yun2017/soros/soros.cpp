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

#include "soros.h"

using namespace iy;



cv::Rect Soros::process(cv::Mat &gray_src, bool is1D /*= true*/, int WinSz /*= 20*/)

{
    cv::setNumThreads(0);
    cv::Rect result(0,0,0,0);

    

    try{

       // saliency map
       cv::Mat saliency = SaliencyMapbyAndoMatrix(gray_src, is1D);

       // integral map
       cv::Mat iMap = calc_integral_image(saliency);

       // find max point with box filter
       cv::Mat sMap(saliency.size(), CV_8UC1); 
       cv::Point cp = find_max_point_with_smooth(iMap, sMap, WinSz);


       // global binzrization
       cv::Mat bMap(saliency.size(), CV_8UC1);
       cv::threshold(sMap, bMap, 50, 255, cv::THRESH_OTSU);

       
       // box detection       
       result = box_detection(bMap, cp);

    }
    catch(cv::Exception &e)
    {
        std::cerr << "cv::Exception: " << std::endl;
        std::cerr << e.what() << std::endl;
    }
    return result;
}

double gmask[7][7] = {{0.0071, 0.0071, 0.0143, 0.0143, 0.0143, 0.0071, 0.0071},
                      {0.0071, 0.0143, 0.0143, 0.0286, 0.0143, 0.0143, 0.0071},
                      {0.0143, 0.0143, 0.0286, 0.0571, 0.0286, 0.0143, 0.0143},
                      {0.0143, 0.0286, 0.0571, 0.1143, 0.0571, 0.0281, 0.0143},
                      {0.0143, 0.0143, 0.0286, 0.0571, 0.0286, 0.0143, 0.0143},
                      {0.0071, 0.0143, 0.0143, 0.0286, 0.0143, 0.0143, 0.0071},
                      {0.0071, 0.0071, 0.0143, 0.0143, 0.0143, 0.0071, 0.0071} };

cv::Mat Soros::SaliencyMapbyAndoMatrix(cv::Mat &src, bool is1D)
{
    const cv::Size imSz = src.size();
    
    cv::Mat result(imSz, CV_8UC1);
    
    double *Ixx = new double[sizeof(double) * imSz.width * imSz.height];
    double *Ixy = new double[sizeof(double) * imSz.width * imSz.height];
    double *Iyy = new double[sizeof(double) * imSz.width * imSz.height];
    double *Cxx = new double[sizeof(double) * imSz.width * imSz.height];
    double *Cxy = new double[sizeof(double) * imSz.width * imSz.height];
    double *Cyy = new double[sizeof(double) * imSz.width * imSz.height];
    
    // edge by sobel
    for(int h = 1; h < imSz.height - 1; h++)
    {
        for(int w = 1; w < imSz.width - 1; w++)
        {
            float dx = (float)src.at<uchar>(h-1,w-1) + 2.0f*src.at<uchar>(h,w-1) + (float)src.at<uchar>(h+1,w-1) - 
                       (float)src.at<uchar>(h-1,w+1) - 2.0f*src.at<uchar>(h,w+1) - (float)src.at<uchar>(h+1,w+1);
                     
            float dy = (float)src.at<uchar>(h-1,w-1) + 2.0f*src.at<uchar>(h-1,w) + (float)src.at<uchar>(h-1,w+1) - 
                       (float)src.at<uchar>(h+1,w-1) - 2.0f*src.at<uchar>(h+1,w) - (float)src.at<uchar>(h+1,w+1);        
            
            Ixx[h*imSz.width + w] = dx*dx;
            Ixy[h*imSz.width + w] = dx*dy;
            Iyy[h*imSz.width + w] = dy*dy;
        }
    } 
    
    // apply gaussian window function
    for(int h = 1; h < imSz.height - 1; h++)
    {
        for(int w = 1; w < imSz.width - 1; w++)
        {
            double C1 = 0;
            double C2 = 0;
            double C3 = 0;
            
            for(int m = 0; m < 7; m++)
            {
                int s = h + m - 4;
                if(s < 0) continue;
                for(int n = 0; n < 7; n++)
                {
                    int k = w + n - 4;
                    if(k < 0) continue;
                    C1 += Ixx[s*imSz.width + k] * gmask[m][n];
                    C2 += Ixy[s*imSz.width + k] * gmask[m][n];
                    C3 += Iyy[s*imSz.width + k] * gmask[m][n];
                }
            }
            
            Cxx[h*imSz.width + w] = C1;
            Cxy[h*imSz.width + w] = C2;
            Cyy[h*imSz.width + w] = C3;
        }
    }
    
    // edge or corner map 
    for(int h = 1; h < imSz.height - 1; h++)
    {
        for(int w = 1; w < imSz.width - 1; w++)
        {
            double Txx = Cxx[h*imSz.width + w];
            double Txy = Cxy[h*imSz.width + w];
            double Tyy = Cyy[h*imSz.width + w];
            
            double m = 0;
            if(is1D)
            {
                m = ((Txx - Tyy)*(Txx - Tyy) + 4 * (Txy * Txy)) / ((Txx + Tyy) * (Txx + Tyy) + 10000); 
            }
            else
            {
                m = (4 * (Txx*Tyy - (Txy*Txy))) / ((Txx + Tyy)*(Txx + Tyy) + 10000);
            }            
            
            m *= 255.0;
            
            result.at<uchar>(h,w) = (m > 255) ? 255 : m; 
        }
    }
    
    delete[]Ixx;
    delete[]Ixy;
    delete[]Iyy;
    delete[]Cxx;
    delete[]Cxy;
    delete[]Cyy;

    return result;
}

cv::Mat Soros::calc_integral_image(cv::Mat &src)
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
       
cv::Point Soros::find_max_point_with_smooth(cv::Mat &src, cv::Mat &smooth_map, int WinSz)
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
        
cv::Rect Soros::box_detection(cv::Mat &src, cv::Point cp)
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