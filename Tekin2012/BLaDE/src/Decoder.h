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
 * @file Barcode Decoding code
 * @author Ender Tekin
 */

#ifndef DECODER_H_
#define DECODER_H_

#include <string>
#include <vector>
#include "ski/BLaDE/Barcode.h"
#include "ski/BLaDE/Symbology.h"
#include <memory>

using namespace std;

/**
 * Barcode decoder class.
 * Decoder methods are responsible for extracting a matrix of energies for a given symbology, where
 * each entry corresponds to the likelihood of a given symbol being a given digit.
 * The symbology used by this decoder then uses this information to perform the final decoding,
 * such as parity checks or joint decoding.
 */
class BarcodeDecoder
{
public:

	/**
	 * Decoder options
	 */
	struct Options
	{
		/** edge threshold for barcode stripe edge detection */
		int edgeThresh;
		/** Approximate fundamental width of extracted slices */
		TUInt fundamentalWidth;
		/** Coefficient to use for the "edginess" when calculating the fixed edge locations */
		double edgePowerCoefficient;
		/** Maximum edge magnitude allowed in the Viterbi*/
		int maxEdgeMagnitude;
		/** Coefficient to use for the variance of the expected locations of fixed edges */
		double edgeFixedLocationVar;
		/** Coefficient to use for the variance of the relative locations of fixed edges */
		double edgeRelativeLocationVar;
		/** Constructor */
		Options():
			edgeThresh(40),
			fundamentalWidth(10),
			edgePowerCoefficient(1),
			maxEdgeMagnitude(200),
			edgeFixedLocationVar(10000),
			edgeRelativeLocationVar(1)
		{};
	};

	/** Result of decoding attempt */
	enum Result {
		CANNOT_DECODE = 0,		///< Means detected barcode is not aligned well enough to attempt decoding
		DECODING_FAILED = -1,	///< Means that the decode attempted decoding, but was not successful
		DECODING_SUCCESSFUL = 1	///< Means that decoding was successful
	};

	/**
	 * Constructor
	 * @param[in] img image to use when decoding barcode
	 * @param[in] symbology to use when decoding. The decoder then takes ownership of the symbology.
	 * @param[in] opts options to use for decoding
	 */
	BarcodeDecoder(const TMatrixUInt8& img, BarcodeSymbology* aSymbology, const Options &opts=Options());

	/**
	 * Destructor
	 */
	~BarcodeDecoder();

	/**
	 * Reads the barcode - main function called for decoding.
	 * @param[in] bc barcode candidate info returned by the detection stage
	 * @return result of attempted decoding attempt
	 */
	Result read(Barcode &bc);

	/**
	 * Name of the symbology used by this decoder
	 */
	inline string symbology() const {return symbology_->name(); };


private:
	/** Decoder options */
	Options opts_;

	/** Possible sweep directions */
	enum SweepDirection {FORWARD = 0, BACKWARD = 1, FINISHED = 2};

	/**
	 * Struct containing information about a detected edge in barcode slice
	 */
	struct DetectedEdge
	{
		/** Constructor */
		inline DetectedEdge(int p, int loc, int mag, int nPrevPos, int nPrevNeg):
				polarity(p), location(loc), magnitude(mag),
				nPreviousPositiveEdges(nPrevPos), nPreviousNegativeEdges(nPrevNeg) {};
		/** Polarity, must be -1 for a light->dark edge, 1 for dark->light edge */
		int polarity;
		/** location of the edge in the slice */
		int location;
		/** magnitude of the detected edge */
		int magnitude;
		/** number of edges with positive polarity before this edge in the barcode slice */
		int nPreviousPositiveEdges;
		/** number of edges with negative polarity before this edge in the barcode slice */
		int nPreviousNegativeEdges;
		/** Index of this edge */
		inline int index() const {return nPreviousPositiveEdges + nPreviousNegativeEdges; };
	};

	/**
	 * Struct containing information about expected barcode symbol boundaries
	 */
	struct SymbolBoundary
	{
		/** Left edge of the symbol */
		int leftEdge;
		/** Right edge of the symbol */
		int rightEdge;
		/** Expected width of the symbol in terms of the fundamental width */
		TUInt width;
		/**
		 * Constructor
		 * @param[in] lEdge location of the left edge of the symbol
		 * @param[in] rEdge location of the right edge of the symbol
		 * @param[in] w width of the symbol in terms of the fundamental width
		 */
		SymbolBoundary(int lEdge=0, int rEdge=0, TUInt w=0) : leftEdge(lEdge), rightEdge(rEdge), width(w) {};
		/**
		 * Fundamental width of the symbol
		 * @return returne the fundamental width of the symbol
		 */
		inline double fundamentalWidth() const {return ((double) rightEdge - leftEdge) / (double) width; };
	};

	/** Grayscale image to estimate the barcode from */
	const TMatrixUInt8& image_;

	/** Symbology used for this detector */
	const std::unique_ptr<BarcodeSymbology> symbology_;

	/** Number of data symbols */
	const TUInt nSymbols_;

	/** Integral barcode slice to be used for symbol estimation */
	vector<int> slice_;

	/**
	 * Performs tests to see whether we should attempt to decode barcode or not.
	 * Decoding is not attempted if it is deemed that the barcode is not properly seen in the image.
	 * @param bc barcode under consideration
	 * @return true if it is determined that the barcode is visible enough to attempt decoding
	 */
	bool shouldAttemptDecoding(const Barcode &bc);

	/**
	 * Extracts the barcode image slice from input image and integrates.
	 * The slice is stretched such that the fundamental width is x.
	 * The extracted slice extends 2x beyond the detected barcode ends
	 * @param[in] aImg grayscale image to extract slice from
	 * @param[in] firstEdge first edge of the barcode candidate.
	 * @param[in] lastEdge last edge of the barcode candidate.
	 */
	void extractIntegralSlice(const TMatrixUInt8& aImg, TPointInt firstEdge, TPointInt lastEdge);

	/**
	 * Extracts edges from the barcode slice
	 * @param[out] extracted edges from the barcode slice
	 */
	void extractEdges(vector<DetectedEdge> &edges);

	/**
	 * Localizes the fixed edges to get accurate symbol boundaries and fundamental width estimates.
	 * @param[out] symbolBoundaries localized symbol boundaries
	 * @return true if an estimate is found, false if not enough edges were determined.
	 */
	bool localizeFixedEdges(vector<SymbolBoundary> &symbolBoundaries);

	/**
	 * Finds which detected edges can be candidates for the fixed edges of the barcode
	 * @param[in] detectedEdges detected edges of the barcode stripe
	 * @param[out] fixedEdgeCandidates fixedEdgeCandidates[i] is a vector of references to the
	 * detected edges that may be fixed edge [i] in the symbology.
	 * @return true if all fixed edges can be matched, false if the detected edges cannot be matched to the symbology.
	 */
	bool getFixedEdgeCandidates(const vector<DetectedEdge> &detectedEdges,
			vector<vector<const DetectedEdge*> > &fixedEdgeCandidates);

	/**
	 * Calculates the fixed edge priors - energies due to the difference of fixed edge candidates from expected absolute locations.
	 * @param[in] fixedEdgeCandidates list of fixed edge candidates as given by getFixedEdgeCandidates()
	 * @param[in] x estimate of the fundamental width to use
	 * @param[out] priors vector of priors for each fixed edge in the symbology.
	 * @param[out] conditionals vector of priors for each fixed edge in the symbology.
	 */
	void calculateFixedEdgeEnergies(const vector<vector<const DetectedEdge*> > &fixedEdgeCandidates,
			double x, vector<vector<TEnergy> > &priors, vector<TMatEnergy> &conditionals);

	/**
	 * Calculate the digit energies
	 * @param[in] dir direction to convolve (in case of backwards barcode)
	 * @param[in] boundaries detected boundaries for the symbols
	 */
	void getDigitEnergies(int dir, const vector<SymbolBoundary> &boundaries);

	/**
	 * Used by convolve(), this function calculates the dot product at a specific place
	 * @param[in] sgn sign of the initial bit - will flip at every pattern edge
	 * @param[in] beginning of the data to calculate the dot-product for
	 * @param[in] pattern pattern to convolve with
	 * @return result of the convolution
	 */
	static int dotProduct(int sgn, const int *data, const vector<TUInt> &pattern);

	/** Matrix to store digit energies, energies_(digit, symbol)*/
	TMatEnergy energies_;

	/** Matrix to store digit convolution values, convolutions_(digit, symbol)*/
	TMatrixInt convolutions_;
};


#endif // DECODER_H_
