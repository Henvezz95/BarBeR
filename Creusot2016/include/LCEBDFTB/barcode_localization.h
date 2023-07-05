#ifndef BARCODE_LOCALIZATION_H
#define BARCODE_LOCALIZATION_H

#include <vector>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>

/**
 * @brief locateBarcode Locates barcode areas on the given picture and uses zbar to decode them.
 * @param image_color Image on which barcodes should be localized/decoded.
 * @param minLineLength Threshold length for under which a line is ignored.
 * @param support_candidates_threshold Threshold for the support value under which contours are ignored.
 * @param delta Number of comparisons (L_i in the paper).
 * @param maxLengthToLineLengthRatio Threshold to ignore contours with a too high length / line-length ratio.
 * @param minLengthToLineLengthRatio Threshold to ignore contours with a too low length / line-length ratio.
 * @param inSegmentXDistance Distance in x-direction for segements within a contour. Used as filter parameter.
 * @param inSegmentYDistance Distance in y-direction for segements within a contour. Used as filter parameter.
 * @return Contours around the barcodes.
 */
std::vector<std::vector<cv::Point> > locateBarcode(cv::Mat image_color,
																									 int minLineLength,
																									 int support_candidates_threshold,
																									 int delta,
																									 int maxLengthToLineLengthRatio,
																									 int minLengthToLineLengthRatio,
																									 int inSegmentXDistance,
																									 int inSegmentYDistance);

/**
 * @brief getLineSegmentsContours Creates contours around the keylines.
 * @param keylines Vector containing all the contours.
 * @param image_lines Image, in which the keylines are drawn for debugging.
 * @param minLineLength Threshold length for under which a line is ignored.
 * @return
 */
std::vector<std::vector<cv::Point>> getLineSegmentsContours(std::vector<cv::line_descriptor::KeyLine> &keylines,
																														cv::Mat& image_lines,
																														int minLineLength);

/**
 * @brief findContainingSegments Finds in all contours the containing segments.
 * @param keylinesInContours Vector which contains for every contour the containing keylines.
 * @param keylines Vector containing all the contours.
 * @param contours Vector containing all the contours.
 * @param contours_size Size of the vector containing all the contours.
 */
void findContainingSegments(std::vector<std::vector<std::shared_ptr<cv::line_descriptor::KeyLine>>> &keylinesInContours,
														std::vector<cv::line_descriptor::KeyLine> keylines,
														std::vector<std::vector<cv::Point>> contours,
														int contours_size);


/**
 * @brief calculateSupportScores Calculates the support scores for all the segments.
 * @param keylinesInContours Vector which contains for every contour the containing keylines.
 * @param support_scores Vector containing for all the contours the support scores for the segments withing the contour.
 * @param keylinesInContours_size Size of the keylinesInCountours vector.
 */
void calculateSupportScores(std::vector<std::vector<std::shared_ptr<cv::line_descriptor::KeyLine>>> &keylinesInContours,
														std::vector<std::vector<int>> &support_scores,
														int keylinesInContours_size);

/**
 * @brief selectSCand Select the segments with the highest support score.
 * @param support_scores Vector containing for all the contours the support scores for the segments withing the contour.
 * @param support_candidates Support scores of the selected segments.
 * @param support_candidates_pos Position of the selected segments.
 * @param keylinesInContours Vector which contains for every contour the containing keylines.
 * @param image_candidates Image the selected segments are drawn in for debugging.
 * @param keylinesInContours_size Size of the keylinesInCountours vector.
 * @param support_candidates_threshold Threshold for the support value under which contours are ignored.
 */
void selectSCand(std::vector<std::vector<int>> &support_scores,
								 std::vector<int> &support_candidates,
								 std::vector<int> &support_candidates_pos,
								 std::vector<std::vector<std::shared_ptr<cv::line_descriptor::KeyLine>>> &keylinesInContours,
								 int keylinesInContours_size,
								 cv::Mat &image_candidates,
								 int support_candidates_threshold);


/**
 * @brief createVectorsOfIntensities Creates vectors of intensities.
 * @param support_candidates Support scores of the selected segments.
 * @param support_candidates_pos Position of the selected segments.
 * @param keylinesInContours Vector which contains for every contour the containing keylines.
 * @param startStopIntensitiesPosition Vector containing the start and stop positions of the intensitiy vector.
 * @param perpendicularLineStartEndPoints Start and end point of the perpendicular line.
 * @param intensities Vector containing the intensities of the five parallel lines for every contour.
 * @param image_greyscale Image from which the intensities are taken.
 * @param image_cols The number of colums of the image.
 * @param image_rows The number of rows of the image.
 * @param intensities_size Size of the intensity vector.
 * @param support_candidates_threshold Threshold for the support value under which contours are ignored.
 * @param deletedContours Binary vector which indicates which contours were deleted (deleted = true, not deleted = false).
 */
void createVectorsOfIntensities(std::vector<int> &support_candidates,
																std::vector<int> &support_candidates_pos,
																std::vector<std::vector<std::shared_ptr<cv::line_descriptor::KeyLine>>> &keylinesInContours,
																std::vector<std::vector<int>> &startStopIntensitiesPosition,
																std::vector<std::vector<cv::Point>> &perpendicularLineStartEndPoints,
																std::vector<std::vector<std::vector<uchar>>> &intensities,
																cv::Mat &image_greyscale,
																int image_cols,
																int image_rows,
																int intensities_size,
																int support_candidates_threshold,
																std::vector<bool> &deletedContours);

/**
 * @brief computePhis Computes the phi values.
 * @param delta Number of comparisons (L_i in the paper).
 * @param intensities Vector containing the intensities of the five parallel lines for every contour.
 * @param intensities_size Size of the intensity vector.
 * @param phis Vector containing the phi values.
 * @param startStopIntensitiesPosition Vector containing the start and stop positions of the intensitiy vector.
 * @param start_barcode_pos Vector containing the start position for every contour/barcode
 * @param end_barcode_pos Vector containing the end position for every contour/barcode
 * @param deletedContours Binary vector which indicates which contours were deleted (deleted = true, not deleted = false).
 */
void computePhis(int delta,
								 std::vector<std::vector<std::vector<uchar>>> &intensities,
								 int intensities_size,
								 std::vector<std::vector<std::vector<int>>> &phis,
								 std::vector<std::vector<int>> &startStopIntensitiesPosition,
								 std::vector<int> &start_barcode_pos,
								 std::vector<int> &end_barcode_pos,
								 std::vector<bool> &deletedContours);


/**
 * @brief calculateBoundingBoxes Calculates the bounding box around the barcodes.
 * @param keylinesInContours_size Size of the keylinesInCountours vector.
 * @param start_barcode_pos Vector containing the start position for every contour/barcode
 * @param end_barcode_pos Vector containing the end position for every contour/barcode
 * @param keylines Vector containing all the contours.
 * @param contours_barcodes Vector containing all the contours.
 * @param perpendicularLineStartEndPoints Start and end point of the perpendicular line.
 * @param image_candidates Image in which the keylines are drawn for debugging.
 * @param deletedContours Binary vector which indicates which contours were deleted (deleted = true, not deleted = false).
 * @param index Index for a contour for debugging.
 * @param maxLengthToLineLengthRatio Threshold to ignore contours with a too high length / line-length ratio.
 * @param minLengthToLineLengthRatio Threshold to ignore contours with a too low length / line-length ratio.
 */
void calculateBoundingBoxes(int keylinesInContours_size,
														std::vector<int> &start_barcode_pos,
														std::vector<int> &end_barcode_pos,
														std::vector<cv::line_descriptor::KeyLine> &keylines,
														std::vector<std::vector<cv::Point>> &contours_barcodes,
														std::vector<std::vector<cv::Point>> &perpendicularLineStartEndPoints,
														cv::Mat &image_candidates,
														std::vector<bool> &deletedContours,
														int index,
														int maxLengthToLineLengthRatio,
														int minLengthToLineLengthRatio);


/**
 * @brief filterContours Filters out contours which cover the same are.
 * @param keylinesInContours_size Size of the keylinesInCountours vector.
 * @param deletedContours Binary vector which indicates which contours were deleted (deleted = true, not deleted = false).
 * @param start_barcode_pos Vector containing the start position for every contour/barcode
 * @param end_barcode_pos Vector containing the end position for every contour/barcode
 * @param keylines Vector containing all the contours.
 * @param support_scores Vector containing for all the contours the support scores for the segments withing the contour.
 * @param contours_barcodes Vector containing the contours of the barcodes.
 * @param inSegmentXDistance Distance in x-direction for segements within a contour. Used as filter parameter.
 * @param inSegmentYDistance Distance in y-direction for segements within a contour. Used as filter parameter.
 */
void filterContours(int keylinesInContours_size,
										std::vector<bool> &deletedContours,
										std::vector<int> &start_barcode_pos,
										std::vector<int> &end_barcode_pos,
										std::vector<cv::line_descriptor::KeyLine> &keylines,
										std::vector<std::vector<int>> &support_scores,
										std::vector<std::vector<cv::Point>> &contours_barcodes,
										int inSegmentXDistance,
										int inSegmentYDistance);


/**
 * @brief contourCenter Calculates the center of a contour
 * @param contour Contour
 * @return Center of the contour
 */
cv::Point contourCenter(const std::vector<cv::Point>& contour);


/**
 * @brief scaleContour Scales a contour according to the scale factor.
 * @param scalingFactor Scaling factor.
 * @param contour Contour to scale.
 * @param image Image used to bound scalced contour to image size.
 * @return Scaled contour.
 */
std::vector<cv::Point> scaleContour(double scalingFactor,
																		const std::vector<cv::Point>& contour,
																		const cv::Mat &image);

/**
 * @brief clamRoiToImage Bounds a region of interes to the size of the image.
 * @param roi Region of interest.
 * @param image Image
 * @return Bounded region of interest.
 */
cv::Rect clamRoiToImage(cv::Rect roi, const cv::Mat& image);

/**
 * @brief decodeBarcode Decodes the barcode with zbar
 * @param keylinesInContours_size Size of the keylinesInCountours vector.
 * @param deletedContours Binary vector which indicates which contours were deleted (deleted = true, not deleted = false).
 * @param contours_barcodes Vector containing the contours of the barcodes.
 * @param image_greyscale Greyscale image.
 * @param image_barcodes Image where the barcode numbers are drawn on.
 */
std::vector<std::string> decodeBarcode(int keylinesInContours_size,
																			 std::vector<bool> &deletedContours,
																			 std::vector<std::vector<cv::Point>> &contours_barcodes,
																			 cv::Mat & image_greyscale,
																			 cv::Mat & image_barcodes);

#endif // BARCODE_LOCALIZATION_H
