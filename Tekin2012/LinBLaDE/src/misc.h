/*
 * misc.h
 *
 *  Created on: Jul 25, 2012
 *      Author: kamyon
 */

#ifndef MISC_H_
#define MISC_H_

#include <cstdio>
#include "ski/log.h"
#include <string>
//#include "ski/cv.hpp"
#include "opencv2/opencv.hpp"
#include "ski/types.h"

namespace Draw
{

typedef struct{
	TUInt8 b, g, r;
} TColor;

static const TColor colorRed = {0, 0, 255};
static const TColor colorGreen = {0, 255, 0};
static const TColor colorBlue = {255, 0, 0};
static const TColor colorWhite = {255, 255, 255};
static const TColor colorBlack = {0, 0, 0};
static const TColor colorLightGray= {200, 200, 200};
static const TColor colorDarkGray = {85, 85, 85};
static const TColor colorYellow = {0, 255, 255};
static const TColor colorCyan = {255, 255, 0};
static const TColor colorMagenta = {255, 0, 255};

/**
 * Draws a line.
 * @param[in] frame image to draw on.
 * @param[in] pt1 one end of the line.
 * @param[in] pt2 other end of the line.
 * @param[in] aColor color of the line.
 * @param[in] thickness in pixels.
 * @param[in] scale that the coordinates were given in.  The result is rendered at scale 0.
 */
void line(cv::Mat &frame, const cv::Point &pt1, const cv::Point &pt2, const TColor &aColor=colorRed, TUInt8 thickness=1, TInt scale=0);

/**
 * Draws a line.
 * @param[in] frame image to draw on.
 * @param[in] pt1 one end of the line.
 * @param[in] pt2 other end of the line.
 * @param[in] aColor brightness of the line.
 * @param[in] thickness in pixels.
 * @param[in] scale that the coordinates were given in.  The result is rendered at scale 0.
 */
void line(cv::Mat &frame, const cv::Point &pt1, const cv::Point &pt2, TUInt8 aIntensity=0, TUInt8 thickness=1, TInt scale=0);

/**
 * Draws a cross.
 * @param[in] frame image to draw on.
 * @param[in] aRect box to draw a cross in.
 * @param[in] aColor color of the cross
 * @param[in] thickness in pixels
 * @param[in] scale that the coordinates were given in.  The result is rendered at scale 0.
 */
void cross(cv::Mat &frame, const cv::Rect &aRect, const TColor &aColor=colorRed, TUInt8 thickness=1, TInt scale=0);

/**
 * Draws a frame.
 * @param[in] frame image to draw on.
 * @param[in] aRect box to draw a frame around.
 * corner[0] is connected to corner[1] which is connected to corner[2], which is
 * connected to corner[3], which is connected back to corner[0].
 * @param[in] aColor color of the frame
 * @param[in] thickness in pixels
 * @param[in] scale that the coordinates were given in.  The result is rendered at scale 0.
 */
void frame(cv::Mat &frame, const cv::Rect &aRect, const TColor& aColor=colorRed, TUInt8 thickness=1, TInt scale=0);

/**
 * Draws text.
 * @param[in] frame image to draw on.
 * @param[in] aPoint anchor point of the text.
 * @param[in] aText text to draw.
 * @param[in] aColor color of the text.
 * @param[in] scale that the coordinates were given in.  The result is rendered at scale 0.
 */
void text(cv::Mat &frame, const cv::Point &aPoint, const std::string &aText, const TColor &aColor=colorRed, TInt scale=0);

/**
 * Draws text on grayscale images.
 * @param[in] frame image to draw on.
 * @param[in] aPoint anchor point of the text.
 * @param[in] aText text to draw.
 * @param[in] aIntensity brightness of the text.
 * @param[in] scale that the coordinates were given in.  The result is rendered at scale 0.
 */
void text(cv::Mat &frame, const cv::Point &aPoint, const std::string &aText, TUInt8 aIntensity, TInt scale=0);

/**
 * Converts a TColor to Scalar.
 * @param[in] aColor TColor to convert
 * @return resulting Scalar.
 */
cv::Scalar convertColor(const TColor &aColor);

/**
 * Converts a Scalar to TColor.
 * @param[in] aColor Scalar to convert
 * @return resulting TColor.
 */
TColor convertColor(const cv::Scalar &aColor);

} //end namespace Draw

#endif /* MISC_H_ */
