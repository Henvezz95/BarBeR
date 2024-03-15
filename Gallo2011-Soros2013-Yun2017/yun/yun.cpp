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

#include <iostream>
#include "yun.h"

using namespace iy;

std::vector<YunCandidate> Yun::process(cv::Mat &gray_src)
{
	cv::setNumThreads(0);
	std::vector<YunCandidate> result;

	try{
		std::vector<YunOrientation> Vmap;
		cv::Mat mMap(gray_src.size(), CV_8UC1);
		cv::Mat oMap = calc_orientation(gray_src, mMap, Vmap);

		// saliency map
		cv::Mat eMap = calc_saliency(oMap, Vmap, pam.localBlockSz);

		cv::Mat iMap = calc_integral_image(eMap);
		cv::Mat sMap = calc_smooth(iMap, pam.winSz);

		cv::Mat bMap; // (sMap.size(), CV_8UC1);
		cv::threshold(sMap, bMap, 50, 255, cv::THRESH_OTSU);

		// search region
		std::vector<YunLabel> blob = ccl(bMap, oMap, Vmap);

		// candidate
		result = calc_candidate(blob, mMap, oMap);

		//clear
		Vmap.clear();
		blob.clear();
	}
	catch (cv::Exception &e)
	{
		std::cerr << "cv::Exception: " << std::endl;
		std::cerr << e.what() << std::endl;
	}
	return result;
}

cv::Mat Yun::calc_orientation(cv::Mat &src, cv::Mat &mMap, std::vector<YunOrientation> &Vmap)
{
	const cv::Size imSz = src.size();
	const int ExtAng = 2 * NUM_ANG;

	cv::Mat oMap(imSz, CV_8UC1);

	Vmap.resize(NUM_ANG);
	for (int i = 0; i < NUM_ANG; i++) Vmap[i].cnt = 0;

	for (int h = 1; h < imSz.height - 1; h++)
	{
		for (int w = 1; w < imSz.width - 1; w++)
		{
			double dx = (double)src.at<uchar>(h - 1, w - 1) + 2.0 * (double)src.at<uchar>(h, w - 1) + (double)src.at<uchar>(h + 1, w - 1) -
				(double)src.at<uchar>(h - 1, w + 1) - 2.0 * (double)src.at<uchar>(h, w + 1) - (double)src.at<uchar>(h + 1, w + 1);

			double dy = (double)src.at<uchar>(h - 1, w - 1) + 2.0 * (double)src.at<uchar>(h - 1, w) + (double)src.at<uchar>(h - 1, w + 1) -
				(double)src.at<uchar>(h + 1, w - 1) - 2.0 * (double)src.at<uchar>(h + 1, w) - (double)src.at<uchar>(h + 1, w + 1);

			int intensity = std::sqrt(dx*dx + dy*dy);
			mMap.at<uchar>(h, w) = intensity > 255 ? 255 : intensity;

			if (intensity > pam.magT)
			{
				double degree = (double)(std::atan2(dy, dx) + CV_PI) * 180.0 / CV_PI; // 0~360;

				// bin
				int bin = ExtAng * (degree / 360.0);
				if (bin < 0) bin = 0;
				else if (bin >= ExtAng) bin = ExtAng - 1;

				// integration
				if (bin > 17) bin -= 18;

				// plus integration (not include paper)
				if (bin > 16 || bin < 2) bin = 0;
				else if (bin < 5)        bin = 3;
				else if (bin < 8)        bin = 6;
				else if (bin < 11)       bin = 9;
				else if (bin < 14)       bin = 12;
				else                     bin = 15;

				// save
				oMap.at<uchar>(h, w) = bin;
				Vmap[bin].cnt++;
			}
			else
			{
				oMap.at<uchar>(h, w) = 255;
			}
		}
	}

	// check orientation
	for (int i = 0; i < NUM_ANG; i++)
	{
		if (Vmap[i].cnt > 6000)
		{
			Vmap[i].isStrong = true;
		}
		else Vmap[i].isStrong = false;
	}

	return oMap;
}

cv::Mat Yun::calc_saliency(cv::Mat &src, std::vector<YunOrientation> &Vmap, int lbSz)
{
	const cv::Size imSz = src.size();

	cv::Mat sMap(imSz, CV_8UC1); sMap.setTo(0);
	const int nMax = lbSz * lbSz * NUM_ANG;
	const int cBlock = (lbSz / 2) + 1;

	for (int h = cBlock; h < imSz.height - cBlock; h += lbSz)
	{
		for (int w = cBlock; w < imSz.width - cBlock; w += lbSz)
		{
			// step 1 local block histogram (orientation)
			int LocalHisto[NUM_ANG] = { 0 };
			for (int y = h - cBlock; y <= h + cBlock; y++)
			{
				if (y < 0 || y >= imSz.height) continue;
				for (int x = w - cBlock; x <= w + cBlock; x++)
				{
					if (x < 0 || x >= imSz.width) continue;

					uchar bin = src.at<uchar>(y, x);
					if (bin >= NUM_ANG) continue;

					LocalHisto[bin]++;
				}
			}

			// step 2 find max values
			int max_val = 0;
			for (int i = 0; i < NUM_ANG; i++)
			{
				if (LocalHisto[i] > max_val)
					max_val = LocalHisto[i];
			}

			// step 3 entropy
			double pim = 0;
			for (int i = 0; i < NUM_ANG; i++)
			{
				pim += std::abs(LocalHisto[i] - max_val);
			}

			// step 4 check max value
			if (max_val == 0) continue;

			// step 5 normalization
			double npim = pim / nMax;
			uchar ramp_npim = (npim * 255) > 255 ? 255 : (npim * 255);

			if (npim < 0.6) ramp_npim = 0;

			// step 6 set block
			for (int y = h - cBlock; y <= h + cBlock; y++)
			{
				if (y < 0 || y >= imSz.height) continue;
				for (int x = w - cBlock; x <= w + cBlock; x++)
				{
					if (x < 0 || x >= imSz.width) continue;
					sMap.at<uchar>(y, x) = ramp_npim;
				}
			}
		}
	}

	return sMap;
}

cv::Mat Yun::calc_integral_image(cv::Mat &src)
{
	//assert(src.channels() == 1);

	const cv::Size imSz = src.size();
	cv::Mat result(imSz, CV_32FC1);

	// -1st cols
	result.at<float>(0, 0) = src.at<uchar>(0, 0);
	for (int w = 1; w < imSz.width; w++)
		result.at<float>(0, w) = (float)src.at<uchar>(0, w) + result.at<float>(0, w - 1);

	// -other cols
	for (int h = 1; h < imSz.height; h++)
	{
		float sum = 0.0f;
		for (int w = 0; w < imSz.width; w++)
		{
			sum += src.at<uchar>(h, w);
			result.at<float>(h, w) = result.at<float>(h - 1, w) + sum;
		}
	}

	return result;
}

cv::Mat Yun::calc_smooth(cv::Mat &src, int WinSz)
{
	const cv::Size imSz = src.size();

	const int nSize = WinSz * WinSz;
	const int cSize = (WinSz / 2) + 1;
	const int max_height = imSz.height - cSize;
	const int max_width = imSz.width - cSize;
	float mean_max = 0.0f;

	cv::Mat smooth_map(imSz, CV_8UC1);

	for (int h = 0; h < imSz.height; h++)
	{
		int temp_top = h - cSize;
		int ntop = (temp_top > max_height) ? max_height : temp_top;
		int temp_bottom = h + cSize;
		int nbottom = (temp_bottom >= imSz.height - 1) ? imSz.height - 1 : temp_bottom;

		for (int w = 0; w < imSz.width; w++)
		{
			int temp_left = w - cSize;
			int nleft = (temp_left > max_width) ? max_width : temp_left;
			int temp_right = w + cSize;
			int nright = (temp_right >= imSz.width - 1) ? imSz.width - 1 : temp_right;

			// local mean
			float n1 = (nleft > 0 && ntop > 0) ? src.at<float>(ntop, nleft - 1) : 0;
			float n2 = (nleft > 0) ? src.at<float>(nbottom, nleft - 1) : 0;
			float n3 = (ntop > 0) ? src.at<float>(ntop, nright) : 0;

			float sum = src.at<float>(nbottom, nright) - n3 - n2 + n1;
			float mean = sum / nSize;

			smooth_map.at<uchar>(h, w) = (mean > 255) ? 255 : mean;
		}
	}

	return smooth_map;
}

int Yun::push(int *stackx, int *stacky, int arr_size, int vx, int vy, int *top)
{
	if (*top >= arr_size) return(-1);
	(*top)++;
	stackx[*top] = vx;
	stacky[*top] = vy;
	return 1;
}

int Yun::pop(int *stackx, int *stacky, int arr_size, int *vx, int *vy, int *top)
{
	if (*top == 0) return(-1);
	*vx = stackx[*top];
	*vy = stacky[*top];
	(*top)--;
	return 1;
}

std::vector<YunLabel> Yun::ccl(cv::Mat &src, cv::Mat &oMap, std::vector<YunOrientation> &Vmap)
{
	std::vector<YunLabel> result;

	const cv::Size imSz = src.size();
	cv::Mat mask(imSz, CV_8UC1); mask.setTo(0);

	int *stackx = new int[imSz.width * imSz.height];
	int *stacky = new int[imSz.width * imSz.height];

	memset(stackx, 0, imSz.width * imSz.height);
	memset(stacky, 0, imSz.width * imSz.height);

	int tsize = imSz.width * imSz.height;
	int r, c, top, label_id = 0;

	cv::Rect rect;

	for (int h = 1; h < imSz.height - 2; h++)
	{
		for (int w = 1; w < imSz.width - 2; w++)
		{
			// skip pixel
			if (mask.at<uchar>(h, w) != 0 || src.at<uchar>(h, w) < 128) continue;

			//
			r = h;
			c = w;
			rect = cv::Rect(w, h, 0, 0);

			top = 0;
			++label_id; if (label_id > 255) label_id = 1;

			int hist[NUM_ANG] = { 0 };

			while (true)
			{
			re:
				for (int m = r - 1; m <= r + 1; m++)
				{
					for (int n = c - 1; n <= c + 1; n++)
					{
						if ((m < 0) || (m >= imSz.height)) continue;	// height ���� ����
						if ((n < 0) || (n >= imSz.width)) continue;	// width ���� ����

						if (mask.at<uchar>(m, n) != 0 ||
							src.at<uchar>(m, n) < 128) continue;

						int bin = oMap.at<uchar>(m, n);
						if (bin >= NUM_ANG) continue;

						mask.at<uchar>(m, n) = label_id;
						hist[bin]++;

						if (push(stackx, stacky, tsize, m, n, &top) == -1) continue;

						r = m; c = n;
						rect.x = rect.x > c ? c : rect.x;
						rect.y = rect.y > r ? r : rect.y;
						rect.width = rect.width > c ? rect.width : c;
						rect.height = rect.height > r ? rect.height : r;
						goto re;
					}
				}

				if (pop(stackx, stacky, tsize, &r, &c, &top) == -1)
				{
					int width = rect.width - rect.x;
					int height = rect.height - rect.y;

					if (width > 15 && height > 15)
					{
						YunLabel val;

						val.roi = cv::Rect(rect.x, rect.y, width, height);

						int max_val = 0;
						int ori = 255;
						for (int i = 0; i < NUM_ANG; i++)
						{
							if (max_val < hist[i])
							{
								max_val = hist[i];
								ori = i;
							}
						}
						val.max_orientation = ori;

						// check Vmap;
						if (Vmap[ori].isStrong)
						{
							result.push_back(val);
						}
					}

					break;
				}
			}
		}
	}

	delete[]stackx;
	delete[]stacky;

	return result;
}

std::vector<YunCandidate> Yun::calc_candidate(std::vector<YunLabel> &val, cv::Mat &mMap, cv::Mat &oMap)
{
	std::vector<YunCandidate> result;

	for (std::vector<YunLabel>::iterator it = val.begin(); it < val.end(); it++)
	{
		YunCandidate tmp = sub_candidate(*it, mMap, oMap);
		if (tmp.isBarcode)
		{
			// not include paper
			YunCandidate new_tmp = calc_region_check(tmp, mMap.size());

			if (result.empty())
			{
				result.push_back(new_tmp);
			}
			else
			{
				bool isSave = true;

				cv::Point st = cv::Point(new_tmp.roi.x, new_tmp.roi.y);
				cv::Point et = cv::Point(new_tmp.roi.x + new_tmp.roi.width, new_tmp.roi.y + new_tmp.roi.height);

				for (std::vector<YunCandidate>::iterator rit = result.begin(); rit < result.end(); rit++)
				{
					cv::Point rst = cv::Point(rit->roi.x, rit->roi.y);
					cv::Point ret = cv::Point(rit->roi.x + rit->roi.width, rit->roi.y + rit->roi.height);

					// compare!
					//if (new_tmp.roi.contains(rst) || new_tmp.roi.contains(ret) ||
					//	rit->roi.contains(st) || rit->roi.contains(et))
					//{
					if (((et.x >= rst.x) && (et.x <= ret.x) && (st.y >= rst.y) && (st.y <= ret.y)) ||
						((st.x >= rst.x) && (st.x <= ret.x) && (st.y >= rst.y) && (st.y <= ret.y)) ||
						((et.x >= rst.x) && (et.x <= ret.x) && (et.y >= rst.y) && (et.y <= ret.y)) ||
						((st.x >= rst.x) && (st.x <= ret.x) && (et.y >= rst.y) && (et.y <= ret.y)) ||

						((rst.x >= st.x) && (rst.x <= et.x) && (rst.y >= st.y) && (rst.y <= et.y)) ||
						((ret.x >= st.x) && (ret.x <= et.x) && (rst.y >= st.y) && (rst.y <= et.y)) ||
						((rst.x >= st.x) && (rst.x <= et.x) && (ret.y >= st.y) && (ret.y <= et.y)) ||
						((ret.x >= st.x) && (ret.x <= et.x) && (ret.y >= st.y) && (ret.y <= et.y))
						)
					{
						// x
						if (st.x <= rst.x)  rit->roi.x = st.x;
						if (et.x >= ret.x)  rit->roi.width = et.x;
						else                rit->roi.width = ret.x;

						// y
						if (st.y <= rst.y)  rit->roi.y = st.y;
						if (et.y >= ret.y)  rit->roi.height = et.y;
						else               rit->roi.height = ret.y;

						rit->roi.width -= rit->roi.x;
						rit->roi.height -= rit->roi.y;

						isSave = false;
						break;
					}
				}

				if (isSave) result.push_back(new_tmp);
			}
		}
	}

	return result;
}

YunCandidate Yun::sub_candidate(YunLabel val, cv::Mat &mMap, cv::Mat &oMap)
{
	YunCandidate result;

	cv::Rect roi = val.roi;
	cv::Size imSz = mMap.size();

	// center point
	cv::Point_<double> cPt = cv::Point(roi.x + (roi.width / 2), roi.y + (roi.height / 2));

	// check dir
	double theta;
	theta = (CV_PI / NUM_ANG) * (val.max_orientation);

	// step
	cv::Point_<double> step(std::cos(theta), std::sin(theta));

	//
	static cv::Rect limit_area(10, 10, imSz.width - 20, imSz.height - 20);
	cv::Rect_<int> imgRect(limit_area);

	int Nedge = 0;
	result.roi = roi;
	result.orientation = val.max_orientation;

	// starting from a Point in the middle
	// Extend in both directions to find the extend
	for (int dir = 0; dir < 2; dir++)
	{
		int dist = 0;
		cv::Point_<double> curPt = cPt;
		cv::Point lastEdge = curPt;

		// change directions
		if (dir == 1) step *= -1.0;

		while (imgRect.contains(curPt))
		{
			curPt += step;

			// line check
			if (mMap.at<uchar>(curPt) > pam.magT)
			{
				if (oMap.at<uchar>(curPt) == val.max_orientation)
				{
					lastEdge = curPt;
					dist = 0;
					Nedge++;
				}
				else if (Nedge > 0)
				{
					dist++;
					Nedge--;
				}
			}
			else if (Nedge > 0) dist++;

			if (dist > 7)
			{
				if (dir == 1)
					result.last_pt = lastEdge;
				else
					result.first_pt = lastEdge;

				break;
			}
		}

		// post processing
		if (dir == 0)
		{
			if (result.first_pt.x == 0 && result.first_pt.y == 0)
				result.first_pt = lastEdge;
		}
		else
		{
			if (result.last_pt.x == 0 && result.last_pt.y == 0)
				result.last_pt = lastEdge;
		}
	}

	int edge_density = cv::norm(result.first_pt - result.last_pt);

	if (Nedge > std::max(pam.minEdgeT, (int)(edge_density * pam.minDensityEdgeT)))
		result.isBarcode = true;
	else result.isBarcode = false;

	return result;
}

YunCandidate Yun::calc_region_check(YunCandidate val, cv::Size imSz)
{
	YunCandidate new_val = val;

	int margin = 1;

	cv::Point st = cv::Point(val.roi.x, val.roi.y);
	cv::Point et = cv::Point(val.roi.x + val.roi.width, val.roi.y + val.roi.height);

	// st
	if (st.y >= val.last_pt.y)        new_val.roi.y = val.last_pt.y - margin;
	else if (st.y >= val.first_pt.y)  new_val.roi.y = val.first_pt.y - margin;
	else                              new_val.roi.y = st.y - margin;

	if (new_val.roi.y < 0) new_val.roi.y = 0;

	if (st.x >= val.last_pt.x)        new_val.roi.x = val.last_pt.x - margin;
	else if (st.x >= val.first_pt.x)  new_val.roi.x = val.first_pt.x - margin;
	else                             new_val.roi.x = st.x - margin;

	if (new_val.roi.x < 0) new_val.roi.x = 0;

	// et
	if (et.y <= val.last_pt.y)        new_val.roi.height = val.last_pt.y + margin;
	else if (et.y <= val.first_pt.y)  new_val.roi.height = val.first_pt.y + margin;
	else                              new_val.roi.height = et.y + margin;

	if (new_val.roi.height >= imSz.height) new_val.roi.height = imSz.height - 1;

	if (et.x <= val.last_pt.x)        new_val.roi.width = val.last_pt.x + margin;
	else if (et.x <= val.first_pt.x)  new_val.roi.width = val.first_pt.x + margin;
	else                              new_val.roi.width = et.x + margin;

	if (new_val.roi.width >= imSz.width) new_val.roi.width = imSz.width - 1;

	new_val.roi.width -= new_val.roi.x;
	new_val.roi.height -= new_val.roi.y;

	return new_val;
}