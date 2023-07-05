/*
Copyright (c) 2012, The Smith-Kettlewell Eye Research Institute
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the The Smith-Kettlewell Eye Research Institute nor
      the names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE SMITH-KETTLEWELL EYE RESEARCH INSTITUTE BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * cv_wrappers.hpp
 *
 *  Created on: Oct 4, 2012
 *      Author: kamyon
 */

#ifndef CV_WRAPPERS_HPP_
#define CV_WRAPPERS_HPP_

#include <opencv2/core/core.hpp>
#include "ski/cv_types.hpp"

namespace ski
{

template <typename T>
cv::Point_<T> cast(const ski::TPoint2<T> &aPt)
{
	return cv::Point_<T>(aPt.x, aPt.y);
};

template <typename T>
ski::TPoint2<T> cast(const cv::Point_<T> &aPt)
{
	return ski::TPoint2<T>(aPt.x, aPt.y);
};

template <typename T>
cv::Size_<T> cast(const ski::TSize<T> &aSz)
{
	return cv::Size_<T>(aSz.width, aSz.height);
};

template <typename T>
ski::TSize<T> cast(const cv::Size_<T> &aSz)
{
	return ski::TSize<T>(aSz.width, aSz.height);
};

template <typename T>
cv::Rect_<T> cast(const ski::TRect<T> &aRect)
{
	return cv::Rect_<T>(aRect.x, aRect.y, aRect.width, aRect.height);
};

template <typename T>
ski::TRect<T> cast(const cv::Rect_<T> &aRect)
{
	return ski::TRect<T>(aRect.x, aRect.y, aRect.width, aRect.height);
};

template <typename T>
cv::Mat_<T> cast(const ski::TMatrix<T> &aMat)
{
	return cv::Mat_<T>(aMat.rows, aMat.cols, aMat.data);
};

template <typename T>
ski::TMatrix<T> cast(const cv::Mat_<T> &aMat)
{
	return ski::TMatrix<T>(aMat.rows, aMat.cols, aMat.data);
};

};

#endif /* CV_WRAPPERS_HPP_ */
