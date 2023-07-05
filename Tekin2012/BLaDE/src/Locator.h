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
 * @file Locator.h
 * Barcode Location Code.
 * @author Ender Tekin
 */

#ifndef BARCODE_LOCATOR_H_
#define BARCODE_LOCATOR_H_

#include <algorithm>
#include <list>
#include "ski/types.h"
#include "algorithms.h"
#include "ski/BLaDE/Barcode.h"

using namespace ski;

/**
 * Class that encapsulates the barcode location algorithm.
 */
class BarcodeLocator
{
public:

	/**
	 * Locator options
	 */
	struct Options
	{
		/** min gradient magnitude threshold */
		TUInt8 gradThresh;
		/** Cell size for the locator to use, each cell has one vote on a barcode orientation.*/
		TUInt cellSize;
		/** Entropy threshold for the barcode verification stage */
		double maxEntropy;
		/** maximum number of votes allowed per Hough bin - to reduce few strong edges overwhelming hough */
		TUInt maxVotesPerBin;
		/** minimum number of votes per orientation*/
		TUInt minVotesPerOrientation;
		/** minimum number of votes per barcode orientation candidate */
		TUInt minVotesPerMode;
		/** minimum edges to declare a segment a barcode segment */
		int minEdgesInBarcode;
		/** minimum edge density to declare a segment a barcode segment in edges/pixel */
		double minEdgeDensityInBarcode;
		/** maximum distance allowed between edges in a barcode */
		int maxDistBtwEdges;
		/** # of orientations being considered */
		TUInt nOrientations;
		/** Scale being used */
		TUInt scale;
		/** Constructor */
		Options():
			gradThresh(20),
			cellSize(16),
			maxEntropy(1.5),
			maxVotesPerBin(20),
			minVotesPerOrientation(300),
			minVotesPerMode(50),
			minEdgesInBarcode(20),
			minEdgeDensityInBarcode(0.2),
			maxDistBtwEdges(5),
			nOrientations(18),
			scale(0)
		{};
	};

	/**
	 * Constructor.
	 * @param[in] img grayscale image to work on
	 * @param[in] opts other locator specific options
	 */
	BarcodeLocator(const TMatrixUInt8 &img, const Options &opts=Options());

	/**
	 * Destructor
	 */
	virtual ~BarcodeLocator();

	/**
	 * Function that locates barcodes - main entry TPointInt for the locator class.
	 * @param[out] list of barcodes found that contains most edges.
	 */
	void locate(BarcodeList &barcodes);

private:
	/** Options used by barcode locator */
	const BarcodeLocator::Options opts_;

	/**
	 * @class Structure that holds the images being used, subsamples input if needed, calculates gradients, etc.
	 * TODO: move to separate file, passing only relevant options
	 */
	class ImageContainer
	{
	private:
		/** Input image */
		const TMatrixUInt8 &original;
		/** scale that we are working on */
		const TUInt scale;
		/** Size of output image */
		const TSizeUInt outputSize;
		/** Subsampled image if scale is greater than zero */
		TMatrixUInt8 scaled;
		/** @f\nabla_i I@f */
		TMatrixInt dI;
		/** @f\nabla_j I@f */
		TMatrixInt dJ;
		/** @f|\nabla\ I| I@f */
		TMatrixUInt8 dMag;
		/** @f\angle\nabla\ I@f */
		TMatrixUInt8 dAng;
		/** Scratch areas for Scharr calculation speedup */
		TMatrixInt tmp1, tmp2;
		/** Matrix containing lookup tables for the gradient orientation given i and j gradients */
		TMatrixUInt8 gradientOrientationMap;
		/** Matrix containing lookup tables for the gradient magnitude given i and j gradients */
		TMatrixUInt8 gradientMagnitudeMap;
		/** Minimum possible value of i gradient */
		static const int MIN_GRAD = -255;
		/** Maximum possible value if j gradient */
		static const int MAX_GRAD = 255;
	public:
		/**
		 * Constructor
		 */
		ImageContainer(const TMatrixUInt8 &input, const Options &opts);

		/**
		 * Calculates the scaled image if needed, recalculates the gradients, etc.
		 */
		void update();

		/**
		 * Whether image is being subsampled
		 */
		inline bool isSubsampled() const {return (scale > 0); };

		/**
		 * Size of output images used by the locator
		 * @return size of gradient images returned by the locator
		 */
		inline TSizeUInt size() const {return outputSize; };

		/**
		 * Returns a reference to the image used for processing
		 * @return the image that is being used for gradient calculations
		 */
		inline const TMatrixUInt8& get() const {return (isSubsampled() ? original : scaled); };

		/**
		 * Magnitude image
		 * @return image of scaled gradient magnitude
		 */
		inline const TMatrixUInt8& magnitudes() const {return dMag; };

		/**
		 * Orientation image
		 * @return image of scaled gradient orientations
		 */
		inline const TMatrixUInt8& orientations() const {return dAng; };

	private:
		/**
		 * Subsamples image if needed
		 */
		static void subsample(const TMatrixUInt8 &input, TMatrixUInt8 &output, TUInt scale);

		/**
		 * Prepares the lookup tables and containers for the gradient calculations
		 * @param[in] thresh minimum threshold for a gradient magnitude - anything lower *in magnitude* is suppressed to zero.
		 * @param[in] nOrientations number of gradient orientations levels to quantize
		 */
		void prepareGradientCalculator(TUInt8 thresh, TUInt8 nOrientations);

		/**
		 * Calculates rectangular and polar gradients with angles quantized to a given number of bins.
		 * @param[in] input input to calculate gradients on
		 */
		void calculateGradients(const TMatrixUInt8& input);

		/**
		 * Calculates rectangular gradients using Sobel or Scharr separable operators.
		 * @param[in] img image to calculate the gradients on.
		 * @param[out] iGrad matrix to return i-gradients in.
		 * @param[out] jGrad matrix to return j-gradients in.
		 * @param[in] tmp1 temporary scratch area
		 * @param[in] tmp2 temporary scratch area
		 */
		static void calculateScharrGradients(const TMatrixUInt8 &img, TMatrixInt &iGrad, TMatrixInt &jGrad, TMatrixInt &tmp1, TMatrixInt &tmp2);

		/**
		 * Calculates polar gradients from rectangular gradients, with orientation quantized to 16 bins.
		 * @param[in] iGrad matrix of i-gradients in.
		 * @param[in] jGrad matrix of j-gradients in.
		 * @param[out] absGrad matrix to return gradient magnitudes in - scaled to fit in an 8 bit unsigned integer.
		 * @param[out] angGrad matrix to return gradient angles in.
		 * @param[in] magnitudeLookup lookup map to use to speed up magnitude scaling
		 * @param[in] orientationLookup lookup map to use to speed up orientation quantization
		 */
		static void calculatePolarGradients(const TMatrixInt &iGrad, const TMatrixInt &jGrad,
				TMatrixUInt8 &absGrad, TMatrixUInt8 &angGrad, const TMatrixUInt8 &magnitudeLookup, const TMatrixUInt8 &orientationLookup);
	} image_;

	/**
	 * Structure containing information about barcode candidate segment and decoded barcode.
	 */
	struct BarcodeCandidate
	{
		/** # of alternating edges in this candidate */
		int nEdges;

		/** orientation of this barcode segment */
		int orientation;

		/** first edge of this barcode segment */
		TPointInt firstEdge;

		/** last edge of this barcode segment */
		TPointInt lastEdge;

		/**
		 * Constructor
		 * @param[in] o orientation
		 * @param[in] pt1 one end of the barcode strip
		 * @param[in] pt2 other end of the barcode strip
		 */
		BarcodeCandidate(int o = 0, const TPointInt &pt1=TPointInt(), const TPointInt &pt2=TPointInt() ) :
			nEdges(0),
			orientation(o),
			firstEdge(pt1),
			lastEdge(pt2)
		{};

		/**
		 * returns the width of the barcode segment
		 * @return width of barcode segment
		 */
		inline double width() const
		{
			return norm(lastEdge - firstEdge);
		};

		/**
		 * Smaller than operator for sorting
		 * @return true if this barcode has fewer edges than the one compared to it
		 * TODO: consider alternative sorting methods using an overall measure including feedback scores?
		 */
		inline bool operator<(const BarcodeCandidate &bc) const
		{
			return (nEdges < bc.nEdges);
		};

		/**
		 * For a barcode candidate determined in a given scale, returns the barcode at full scale
		 * @return detected barcode
		 */
		inline Barcode promote(TUInt scale) const
		{
			int multiplier = (1 << scale);
			return Barcode(	firstEdge * multiplier, lastEdge * multiplier);
		}
	};

	typedef std::list<BarcodeCandidate> BarcodeCandidateList;

	/** list of detected barcodes */
	BarcodeCandidateList barcodeCandidates_;

	/**
	 * An image is divided into cells that are deemed part of a barcode or not based on the number
	 * of edge pixels and the histogram of orientations of the edge pixels in the cell
	 */
	class Cell
	{
	public:
		/** boundaries of the cell */
		TRectInt box;
		/** Histogram of orientations for this cell */
		vector<TUInt> orientationHistogram;
		/** Histogram of orientations for this cell */
		vector<TUInt> weightedOrientationHistogram;
	private:
		/** Dominant orientation of this cell, value is set by findDominantOrientation(), which must be run before reading. */
		int dominantOrientation_;
		/** Entropy of this cell, value is set by calculateEntropy(), which must be run before reading. */
		double entropy_;
		/** Number of pixels voting in the orientation histogram */
		TUInt nVoters_;
		/** Maximum entropy allowed */
		double maxEntropy_;
		/** Number of possible orientations */
		TUInt nOrientations_;
	public:
		/**
		 * Constructor
		 * @param[in] aBox cell area.
		 * @param[in] nOrientations number of orientation bins.
		 * @param[in] maximum entropy allowed
		 */
		Cell(const TRectInt &aBox, TUInt nOrientations, double maxEntropy);
		/**
		 * Clears the voters the orientation histogram, and the isSet flag.
		 */
		void reset();
		/**
		 * Adds a new pixel vote
		 * @param[in] orientation orientation of the pixel
		 * @param[in] magnitude of this pixel
		 */
		void addVoter(TUInt8 orientation, TUInt8 magnitude);
		/**
		 * Returns the dominant orientation in this cell and sets the dominantOrientation value.
		 * @return dominant orientation
		 */
		int dominantOrientation();
		/**
		 * Calculates the entropy of this cell, and sets the entropy value.
		 * @return entropy
		 */
		double entropy();
		/**
		 * Returns true is this cell has low entropy
		 */
		inline bool hasLowEntropy() {return ( entropy() < maxEntropy_ ); };
		/**
		 * Returns true is this cell has sufficient number of voters
		 */
		inline bool hasEnoughVoters() const {return ( nVoters_ > ((TUInt) box.area() >> 2) ); };
		/**
		 * True if this cell passes the tests for further consideration
		 * @return true if the cell has both low entropy and sufficient number of voters
		 */
		inline bool shouldBeConsidered() {return hasLowEntropy() && hasEnoughVoters(); };
		/**
		 * Returns the number of voting pixels
		 */
		inline TUInt nVoters() {return nVoters_; };
		/**
		 * Returns the center of the cell
		 */
		inline TPointInt center() {return (box.tl() + box.br()) * .5; };
	};

	/**
	 * Collects votes from each pixel for barcode location/orientation.
	 * @param[out] modes modes of the orientation histogram.
	 */
	void getOrientationCandidates(vector<Vote> &modes);

	/**
	 * Calculates the histograms of non-overlapping cells in the image
	 */
	void calculateCellHistograms();

	/**
	 * Calculates the image orientation histogram from cell histograms.
	 */
	void calculateOrientationHistogram();

	/**
	 * Calculates the modes of the orientation histogram
	 * @param[out] modes modes of the orientation histogram.
	 */
	void findOrientationHistogramModes(vector<Vote> &modes);

	/**
	 * Called by histogram modes, finds the modes using gradient ascent.
	 */
	void ascendModes(const vector<Vote> &votes, vector<Vote> &modes);

	/**
	 * Finds barcode candidates at given orientation candidates.
	 * @param[in] modes modes of the orientation histogram.
	 */
	void getBarcodeCandidates(const vector<Vote> &modes);

	/**
	 * Finds clusters of barcode candidates at a given orientation
	 * @param[in] theta orientation to look for the candidates in
	 * @param[out] candidates returns the candidates for barcode centers
	 */
	void getCandidateCellClusters(double theta, vector<TPointInt> &candidates);

	/**
	 * Scans a segment to see if there is barcode evidence.
	 * @param[out] aBC barcode to save if the segment is indeed a good candidate
	 * @param[in] pt TPointInt to start the scan
	 * @param[in, out] qBegin pointer to where in the scanline we start the scan
	 * @param[in] qEnd pointer to where in the scanline we end the scan
	 * The scan starts from pt, proceeds along qBegin .. qEnd unless a barcode segment is found,
	 * in which case it returns true and qBegin points at the last TPointInt we were in the scanline.
	 * @return true if a viable barcode segment has been found.
	 */
	bool scanSegment(BarcodeCandidate &aBC, const TPointInt &pt);

	/**
	 * Prepares the pixel-to-cell lookup table and cell vector
	 */
	void prepareCells();

	/**
	 * Generates trigonometric lookup tables for the Hough transform
	 */
	void prepareTrigLookups();

	/**
	 * Prepares the scanlines to follow
	 */
	void prepareScanLines();

	/** matrix of cells from the image */
	vector<vector<Cell> > cells_;

	/** Mapping from pixels to cell indices to easily find which cell pixel (i,j) belongs to. */
#ifdef USE_OPENCV	//When using gcc4.7+, use type-alias
	cv::Mat_<Cell*> mapPixelToCell_;
#else
	ski::TMatrix<Cell*> mapPixelToCell_;
#endif

	/**
	 * Lookup table for hough votes, cosLookupTable(i)=i*cos(theta)/dr and sinLookupTable(j)=j*sin(theta)/dr
	 * r(i,j) = cosLookupTable[i] + sinLookupTable[j];
	 */
	TMatrixInt cosLookupTable_, sinLookupTable_;

	/**
	 * Histogram of hough votes over the orientations (column summation of the Hough matrix)
	 */
	vector<TUInt> orientationHistogram_;

	/** Scan lines to sweep. iScanLines_[o] is a scanline to sweep for orientation o */
	vector<vector<TPointInt> > scanLines_;

	/** Acceptable angles for a given orientation */
	TMatrixBool isAcceptable_;

};

#endif //BARCODE_LOCATOR_H_
