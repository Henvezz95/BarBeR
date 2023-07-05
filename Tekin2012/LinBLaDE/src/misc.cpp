/*
 * misc.cpp
 *
 *  Created on: Jul 25, 2012
 *      Author: kamyon
 */

#include "misc.h"

using namespace cv;

void Draw::line(Mat &frame, const Point &pt1, const Point &pt2, const TColor &aColor/*=colorRed*/, TUInt8 thickness/*=1*/, TInt scale/*=0*/){
	Point ptA(pt1.x << scale, pt1.y << scale), ptB(pt2.x << scale, pt2.y << scale);
	cv::line(frame, ptA, ptB, convertColor(aColor), thickness);
}

void Draw::line(Mat &frame, const Point &pt1, const Point &pt2, TUInt8 aIntensity/*=0*/, TUInt8 thickness/*=1*/, TInt scale/*=0*/){
	Point ptA(pt1.x << scale, pt1.y << scale), ptB(pt2.x << scale, pt2.y << scale);
	cv::line(frame, ptA, ptB, aIntensity, thickness);
}

void Draw::cross(Mat &frame, const Rect &aRect, const TColor &aColor /*=colorRed*/, TUInt8 thickness /*=1*/, TInt scale /*=0*/){
	Draw::line(frame, aRect.tl(), aRect.br(), aColor, thickness, scale);
	Draw::line(frame, Point(aRect.tl().x, aRect.br().y), Point(aRect.br().x, aRect.tl().y), aColor, thickness, scale);
}

void Draw::frame(Mat &frame, const Rect &aRect, const TColor &aColor/*=colorRed*/, TUInt8 thickness/*=1*/, TInt scale/*=0*/){
	Point ptA( aRect.tl().x << scale, aRect.tl().y << scale), ptB(aRect.br().x << scale, aRect.br().y << scale);
    cv::rectangle(frame, ptA, ptB, convertColor(aColor), thickness);
}

void Draw::text(Mat &frame, const Point &aPt, const std::string &aText, const TColor &aColor/*=colorRed*/, TInt scale/*=0*/){
	Point anchorPt(aPt.x << scale, aPt.y << scale);
	cv::putText(frame, aText, anchorPt, FONT_HERSHEY_PLAIN, 1, convertColor(aColor));
}

void Draw::text(Mat &frame, const Point &aPt, const std::string &aText, TUInt8 aIntensity/*=colorRed*/, TInt scale/*=0*/){
	Point anchorPt(aPt.x << scale, aPt.y << scale);
	cv::putText(frame, aText, anchorPt, FONT_HERSHEY_PLAIN, 1, aIntensity);
}

Scalar Draw::convertColor(const Draw::TColor &aColor) {
	return Scalar(aColor.b, aColor.g, aColor.r);
}

Draw::TColor Draw::convertColor(const Scalar &aColor) {
	TColor color = { (TUInt8) aColor.val[0], (TUInt8) aColor.val[1], (TUInt8) aColor.val[2] };
	return color;
}
