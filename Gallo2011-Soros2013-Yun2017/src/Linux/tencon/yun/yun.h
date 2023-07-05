/*
*  Copyright 2014-2017 Inyong Yun (Sungkyunkwan University)
*
*  Created on: 2017. 11. 24. (Rev. 2.0)
*      Author: Inyong Yun
*        type: c/c++
*
*   Reference paper: 
*   I.Y. Yun & J.K. Kim, (2017) "Vision-Based 1D Barcode Localization Method for Scale and Rotation Invariant."
*   Proc. of the 2017 IEEE Region 10 Conference (TENCON), Malaysia, Nov 5-8, 2204-2208
*
*   etc: not include barcode recognition process.
*        only localization method! *
*/
#ifndef YUN
#define YUN
#include <opencv2/opencv.hpp>
#include <vector>

#define NUM_ANG  18

namespace iy{
	typedef struct
	{
		int cnt;
		bool isStrong;
	} YunOrientation;

	typedef struct
	{
		cv::Rect roi;
		int max_orientation;
	} YunLabel;

	typedef struct 
	{
		cv::Point last_pt, first_pt;
		cv::Rect roi;
		int orientation;
		bool isBarcode;
	} YunCandidate;
	
	typedef struct
	{
		int magT;
		int winSz;
		int minEdgeT;
		int localBlockSz;
		double minDensityEdgeT;
	} YunParams;

	class Yun{
	private:
		// process parameter
		YunParams pam;

		cv::Mat calc_orientation(cv::Mat &src, cv::Mat &mMap, std::vector<YunOrientation> &Vmap);
		cv::Mat calc_saliency(cv::Mat &src, std::vector<YunOrientation> &Vmap, int lbSz);
		cv::Mat calc_integral_image(cv::Mat &src);
		cv::Mat calc_smooth(cv::Mat &src, int WinSz);
		int push(int *stackx, int *stacky, int arr_size, int vx, int vy, int *top);
		int pop(int *stackx, int *stacky, int arr_size, int *vx, int *vy, int *top);
		std::vector<YunLabel> ccl(cv::Mat &src, cv::Mat &oMap, std::vector<YunOrientation> &Vmap);
		std::vector<YunCandidate> calc_candidate(std::vector<YunLabel> &val, cv::Mat &mMap, cv::Mat &oMap);
		YunCandidate sub_candidate(YunLabel val, cv::Mat &mMap, cv::Mat &oMap);
		YunCandidate calc_region_check(YunCandidate val, cv::Size imSz);

	public:
		Yun() {
			// init value
			pam.magT = 30;
			pam.winSz = 25;
			pam.minEdgeT = 30;
			pam.localBlockSz = 15;
			pam.minDensityEdgeT = 0.3;
		}
		~Yun() {}

		std::vector<YunCandidate> process(cv::Mat &gray_src);
		std::vector<YunCandidate> process(cv::Mat &gray_src, YunParams pams)
		{
			pam = pams;
			std::vector<YunCandidate> result = process(gray_src);
			return result;
		};
	};
}
#endif