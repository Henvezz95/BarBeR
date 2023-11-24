#include <vector>
#include <memory>
#include <string>
#include <stdlib.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include "gallo/gallo.h"
#include "soros/soros.h"
#include "yun/yun.h"
#include "wrapper.h"

iy::Gallo mGallo;
iy::Soros mSoros;
iy::Yun mYun;

void galloProcess(int* result, unsigned char* img_color, int h, int w, int WinSz){
  cv::Mat image_color(h, w, CV_8UC3, img_color);
  cv::Mat image_greyscale;
  cv::cvtColor(image_color, image_greyscale, cv::COLOR_BGR2GRAY);
  cv::Rect g_rt = mGallo.process(image_greyscale, WinSz);
  result[0] = g_rt.x;
  result[1] = g_rt.y;
  result[2] = g_rt.width;
  result[3] = g_rt.height;
}


void sorosProcess(int* result, unsigned char* img_color, int h, int w, bool is1D, int WinSz){
  cv::Mat image_color(h, w, CV_8UC3, img_color);
  cv::Mat image_greyscale;
  cv::cvtColor(image_color, image_greyscale, cv::COLOR_BGR2GRAY);
  cv::Rect g_rt = mSoros.process(image_greyscale, is1D=is1D, WinSz=WinSz);
  result[0] = g_rt.x;
  result[1] = g_rt.y;
  result[2] = g_rt.width;
  result[3] = g_rt.height;
}


void yunProcess(int* result, int* num_results, unsigned char* img_color, int h, int w){
  *num_results = 0;
  cv::Mat image_color(h, w, CV_8UC3, img_color);
  cv::Mat image_greyscale;
  cv::cvtColor(image_color, image_greyscale, cv::COLOR_BGR2GRAY);
  std::vector<iy::YunCandidate> list_barcode = mYun.process(image_greyscale);
	if (!list_barcode.empty())
	{
		for (std::vector<iy::YunCandidate>::iterator it = list_barcode.begin(); it < list_barcode.end(); it++)
		{
			if (it->isBarcode)
			{
				cv::Rect y_rt = it->roi;
        result[0+(*num_results)*4] = y_rt.x;
        result[1+(*num_results)*4] = y_rt.y;
        result[2+(*num_results)*4] = y_rt.width;
        result[3+(*num_results)*4] = y_rt.height;
        (*num_results)++;
			}
		}

		list_barcode.clear();
	}



}