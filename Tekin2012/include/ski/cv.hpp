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

/**
 * @file Computer vision type declarations.
 * @author Ender Tekin
 */

#ifndef SKI_CV_H_
#define SKI_CV_H_

#define GCC_VERSION (__GNUC__ * 10000  + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#define CAN_USE_TEMPLATE_ALIAS false //(GCC_VERSION >= 40700) //template aliases are implemented in gcc4.7+

#include "ski/types.h"
#include <vector>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
//simple cv structures
typedef cv::Point_<int> TPointInt;
typedef cv::Point_<TUInt> TPointUInt;
typedef cv::Point_<double> TPointDouble;
typedef cv::Size_<int> TSizeInt;
typedef cv::Size_<TUInt> TSizeUInt;
typedef cv::Size_<double> TSizeDouble;
typedef cv::Rect_<int> TRectInt;
typedef cv::Rect_<TUInt> TRectUInt;
typedef cv::Rect_<double> TRectDouble;
//matrices
typedef cv::Mat_<int> TMatrixInt;
typedef cv::Mat_<TUInt> TMatrixUInt;
typedef cv::Mat_<TUInt8> TMatrixUInt8;
typedef cv::Mat_<float> TMatrixFloat;
typedef cv::Mat_<double> TMatrixDouble;
typedef cv::Mat_<bool> TMatrixBool;
#if CAN_USE_TEMPLATE_ALIAS
//matrix template alias
template <typename T> using TMat = cv::Mat_<T>;
#endif
#else
#include "ski/cv_types.hpp"
typedef ski::TPoint2<int> TPointInt;
typedef ski::TPoint2<TUInt> TPointUInt;
typedef ski::TPoint2<double> TPointDouble;
typedef ski::TSize<int> TSizeInt;
typedef ski::TSize<TUInt> TSizeUInt;
typedef ski::TSize<double> TSizeDouble;
typedef ski::TRect<int> TRectInt;
typedef ski::TRect<TUInt> TRectUInt;
typedef ski::TRect<double> TRectDouble;
//matrices
typedef ski::TMatrix<int> TMatrixInt;
typedef ski::TMatrix<TUInt> TMatrixUInt;
typedef ski::TMatrix<TUInt8> TMatrixUInt8;
typedef ski::TMatrix<float> TMatrixFloat;
typedef ski::TMatrix<double> TMatrixDouble;
typedef ski::TMatrix<bool> TMatrixBool;
#if CAN_USE_TEMPLATE_ALIAS
//matrix template alias
template <typename T> using TMat = class ski::TMatrix<T>;
#endif
#endif

//arrays
typedef std::vector<int> TArrayInt;
typedef std::vector<TUInt> TArrayUInt;
typedef std::vector<TUInt8> TArrayUInt8;
typedef std::vector<float> TArrayFloat;
typedef std::vector<double> TArrayDouble;
typedef std::vector<bool> TArrayBool;

#endif // SKI_CV_H_
