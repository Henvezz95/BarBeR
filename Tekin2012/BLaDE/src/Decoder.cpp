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

#include "Decoder.h"
#include "ski/types.h"
#include "ski/viterbi.h"
#include "ski/log.h"
#include "algorithms.h"

BarcodeDecoder::BarcodeDecoder(const TMatrixUInt8 &img, BarcodeSymbology *aSymbology, const Options &opts/*=Options()*/):
	opts_(opts),
	image_(img),
	symbology_(aSymbology),
	nSymbols_(symbology_->nDataSymbols()),
	slice_((symbology_->width() + 4) * opts.fundamentalWidth),
	energies_(10, nSymbols_),
	convolutions_(10, nSymbols_)
{
	LOGD("Decoder created for symbology %s (%u symbols of total width %u, with %u edges)\n",
			symbology_->name(), symbology_->nDataSymbols(), symbology_->width(), symbology_->nTotalEdges());
}

BarcodeDecoder::~BarcodeDecoder()
{
}

BarcodeDecoder::Result BarcodeDecoder::read(Barcode &bc)
{
	try
	{
		if (!shouldAttemptDecoding(bc))
			return CANNOT_DECODE;
		//At this TPointInt, we have an approximately oriented barcode, extract detection slice
		extractIntegralSlice(image_, bc.firstEdge, bc.lastEdge);
		//Localize the fixed edges = symbol boundaries
		vector<SymbolBoundary> boundaries;
		if (localizeFixedEdges(boundaries))
		{
			string estimatedBarcode;
			//Try to decode
			for (int dir = FORWARD; dir < FINISHED; dir++) //for each direction
			{
				//Convolve with the patterns to get energies
				getDigitEnergies(dir, boundaries);
				//Estimate barcode with this symbology
				LOGD("Attempting estimation of barcode as %s in the %s direction:\n", symbology_->name(), (dir == FORWARD ? "forward" : "backward"));
				estimatedBarcode = symbology_->estimate(energies_);
				//if correct estimate, quit and return the estimate
				if (!estimatedBarcode.empty())
				{
					bc.estimate = estimatedBarcode;
					bc.symbology = symbology_->name();
					return DECODING_SUCCESSFUL;
				}
			}
		}
		//Attempted but failed decoding
		return DECODING_FAILED;
	}
	catch (const exception &aErr)
	{
		LOGD("Decoder Error: %s\n", aErr.what());
		throw aErr;
	}
	return CANNOT_DECODE;
}

bool BarcodeDecoder::shouldAttemptDecoding(const Barcode &bc)
{
	TUInt M = image_.rows, N = image_.cols;
	TPointDouble d = bc.lastEdge - bc.firstEdge;
	LOGD("Detecting whether barcode (%d,%d)-(%d,%d) is %f degrees should be decoded\n", bc.firstEdge.x, bc.firstEdge.y, bc.lastEdge.x, bc.lastEdge.y, atan2(d.y, d.x) * 180.0 / ski::PI);
	//double angle = atan2(d.y, d.x);
	//double imWidth = .8 * min(N / std::abs(std::cos(angle)), M / std::abs(std::sin(angle)) );
	//double w = norm(d), maxWidth = .8 * imWidth, minWidth = .4 * imWidth;
	//bool isTooSmall = (w < minWidth), isTooBig = (w > maxWidth);
	int w = abs(d.x), h = abs(d.y);
	bool isTooSmall = ( (w < .4 * N) && (h < .4 * M) ), isTooBig = ( (w > .8 * N) || (h > .8 * M) );

	int minDist = min(M, N) / 20; //how far the edges should be from the edge of the image
	int leftDist = min(bc.firstEdge.x, bc.lastEdge.x), rightDist = N - max(bc.firstEdge.x, bc.lastEdge.x);
	int topDist = min(bc.firstEdge.y, bc.lastEdge.y), botDist = M - max(bc.firstEdge.y, bc.lastEdge.y);
	bool isTooCloseToEdges = (leftDist < minDist) || (rightDist < minDist) || (topDist < minDist) || (botDist < minDist);
	if (isTooSmall)
	{
		//LOGD("Barcode is too small (%f < %f)\n", w, minWidth);
		LOGD("Barcode is too small (%f)\n", norm(d));
		return false;
	}
	else if (isTooBig)
	{
		LOGD("Barcode is too big (%f)\n", norm(d));
		return false;
	}
	else if (isTooCloseToEdges)
	{
		LOGD("Barcode is too close to edges (%d < %d)\n", max(max(leftDist, rightDist), max(topDist, botDist)), minDist);
		return false;
	}
	else
	{
		LOGD("Barcode is sufficiently resolved to attempt decoding\n");
		return true;
	}
	//return ( (w < maxWidth) && (w > minWidth) && (leftDist > minDist) && (rightDist > minDist) && (topDist > minDist) && (botDist > minDist) );
}

void BarcodeDecoder::extractIntegralSlice(const TMatrixUInt8& aImg, TPointInt firstEdge, TPointInt lastEdge)
{
	//for each TPointInt on this slice
	double fundamentalWidth = norm(lastEdge - firstEdge) / symbology_->width();
	double scaling = (double) opts_.fundamentalWidth / fundamentalWidth;
	TPointDouble d = lastEdge - firstEdge;
	double theta = atan2(d.y, d.x);
	TPointInt offset(2 * cos(theta) * fundamentalWidth, 2 * sin(theta) * fundamentalWidth);
	//extend 2X past the first and last edges.
	firstEdge -= offset;
	lastEdge += offset;
	//Recalculate step and scaling to account for quantization
	d = lastEdge - firstEdge;
	theta = atan2(d.y, d.x);
	scaling = (double) slice_.size() / norm(d);
	TPointDouble step(cos(theta) / scaling, sin(theta) / scaling);
	TPointDouble pt((double) firstEdge.x, (double) firstEdge.y);
	TPointInt qt; //quantized version of pt
	slice_.front() = aImg(firstEdge);
	slice_.back() = aImg(lastEdge);
	vector<int>::iterator s = slice_.begin();
	for (s++; s != slice_.end() - 1; s++)
	{
		pt += step;
		qt.x = floor(pt.x);
		qt.y = floor(pt.y);
		double dx = pt.x - qt.x;
		double dy = pt.y - qt.y;
		//Interpolate & integrate
		*s = (1-dy) * ((1-dx) * aImg(qt) + dx * aImg(qt.y, qt.x+1)) \
				+ dy * ((1-dx) * aImg(qt.y+1, qt.x) + dx * aImg(qt.y+1, qt.x+1)) \
				+ *(s-1);
	}
	//Last TPointInt. Now, s -> slice_.end() - 1 = slice.back();
	*s += *(s-1);
}

void BarcodeDecoder::extractEdges(vector<DetectedEdge> &edges)
{
	edges.clear();
	static const TUInt width = opts_.fundamentalWidth / 2; //for edge filter
	TUInt nPrevPos = 0, nPrevNeg = 0;
	int ePrev = 0, e, eNext;
	vector<int>::iterator i = slice_.begin() + width;
	e = *(i + width) + *(i - width) - 2 * (*i);

	for (i++; i != slice_.end() - width - 1; i++)
	{
		eNext = *(i + width) + *(i - width) - 2 * (*i);
		if ( (e > opts_.edgeThresh) && (e > ePrev) && (e >= eNext ) ) 			//if it's a local max
			edges.push_back( DetectedEdge(1, i - slice_.begin() - 1, e, nPrevPos++, nPrevNeg) );
		else if ( (e < -opts_.edgeThresh) && (e < ePrev) && (e <= eNext) ) 		//if it's a local min
			edges.push_back( DetectedEdge(-1, i - slice_.begin() - 1, -e, nPrevPos, nPrevNeg++) );
		ePrev = e;
		e = eNext;
	}
}

bool BarcodeDecoder::localizeFixedEdges(vector<SymbolBoundary> &symbolBoundaries)
{
	//extract edges from barcode strip
	static vector<DetectedEdge> detectedEdges;
	detectedEdges.reserve(100);	//reserve space assuming no more than 100 edges detected (if more, the array may move)
	extractEdges(detectedEdges);
	//get edge candidates
	static TUInt nFixedEdges = symbology_->nFixedEdges();
	static vector<vector<TEnergy> > priors(nFixedEdges);
	static vector<TMatEnergy> conditionals(nFixedEdges-1, TMatEnergy(0,0));
	//Get list of fixed edge candidates among detected edges
	static vector<vector<const DetectedEdge*> > fixedEdgeCandidates(nFixedEdges); //fixedEdgeCandidates[i] has pointers to detected edges that may be fixed edge i
	if (!getFixedEdgeCandidates(detectedEdges, fixedEdgeCandidates))
		return false;
	//Determine fixed edge locations
	double xInit, x = ((double) (fixedEdgeCandidates.back().back()->location - fixedEdgeCandidates.front().front()->location) ) / symbology_->width();
	//Resize the prior and conditional matrices
	for (TUInt n = 0; n < nFixedEdges; n++)
	{
		TUInt M = fixedEdgeCandidates[n].size();
		priors[n].resize(M);
		if (n < nFixedEdges-1)
		{
			TUInt N = fixedEdgeCandidates[n+1].size();
			conditionals[n] = TMatEnergy(M, N);
		}
	}
	//Prepare the viterbi
	Viterbi<TEnergy> V(priors, conditionals);
	do
	{
		LOGD("x estimated = %f\n", x);
		xInit = x;
		//Calculate energies
		calculateFixedEdgeEnergies(fixedEdgeCandidates, x, priors, conditionals);
		//Perform the Viterbi
		try
		{
			V.solve();
			vector<int> &bestFitEdges = V.solutions[0].sequence;
			x = ((double) (fixedEdgeCandidates.back()[bestFitEdges.back()]->location - fixedEdgeCandidates.front()[bestFitEdges.front()]->location )) / symbology_->width();
		}
		catch (exception &aErr)
		{
			LOGE("Error in fixed edge calculation: %s\n", aErr.what());
			throw;
		}
	}
	while (abs(x-xInit) > 0.01 * x);	//repeat until convergence (to 1%) of the fundamental width.
	//Return the symbol boundaries:
	symbolBoundaries.resize(nSymbols_);
	vector<int> &bestFitEdges = V.solutions[0].sequence;
	for (TUInt s = 0, e = 0; s < nSymbols_; s++)
	{
		const BarcodeSymbology::Symbol* aSymbol = symbology_->getDataSymbol(s);
		symbolBoundaries[s].width = aSymbol->width;
		const BarcodeSymbology::Edge* aEdge = aSymbol->leftEdge();
		//Find the corresponding fixed edge index
		while (symbology_->getFixedEdge(e) != aEdge)
			e++;
		symbolBoundaries[s].leftEdge = fixedEdgeCandidates[e][bestFitEdges[e]]->location;
		aEdge = aSymbol->rightEdge();
		//Find the corresponding fixed edge index
		while (symbology_->getFixedEdge(e) != aEdge)
			e++;
		symbolBoundaries[s].rightEdge = fixedEdgeCandidates[e][bestFitEdges[e]]->location;
	}
	return true;
}

void BarcodeDecoder::calculateFixedEdgeEnergies(const vector<vector<const DetectedEdge*> > &fixedEdgeCandidates,
		double x, vector<vector<TEnergy> > &priors, vector<TMatEnergy> &conditionals)
{
	TEnergy energy;
	const BarcodeSymbology::Edge *pEdge, *pNextEdge;
	static const double coeffPrior = 1 / opts_.edgeFixedLocationVar, coeffConditional = 1 / opts_.edgeRelativeLocationVar;
	//Priors
	static const TUInt nFixedEdges = symbology_->nFixedEdges();
	for (TUInt n = 0; n < nFixedEdges; n++)
	{
		pEdge = symbology_->getFixedEdge(n);
		double expectedEdgeLocation = 1 + pEdge->location;
		TUInt M = fixedEdgeCandidates[n].size();
		//priors[n].resize(M);
		TEnergy *pPrior = &(priors[n].front());
		for (TUInt i = 0; i < M; i++)
		{
			const DetectedEdge *pE = fixedEdgeCandidates[n][i];
			//"edginess" factor
			energy = opts_.edgePowerCoefficient * max(opts_.maxEdgeMagnitude - pE->magnitude, 0);
			//offset from expected location
			double edgeLocation = pE->location / x;
			double distanceFromExpected = abs(expectedEdgeLocation - edgeLocation);	//normalize by x, and account for the +-x gap on each side of the slice
			energy += coeffPrior * distanceFromExpected * distanceFromExpected;
			pPrior[i] = energy;
		}
	}
	//Conditionals
	for (TUInt n = 0; n < nFixedEdges-1; n++)
	{
		pEdge = symbology_->getFixedEdge(n);
		pNextEdge = symbology_->getFixedEdge(n+1);
		double expectedInterEdgeDistance = pNextEdge->location - pEdge->location;
		TUInt M = fixedEdgeCandidates[n].size(), N = fixedEdgeCandidates[n+1].size();
		conditionals[n] = TMatEnergy(M, N);
		for (TUInt i = 0; i < M; i++)
		{
			const DetectedEdge *pE = fixedEdgeCandidates[n][i];
			TEnergy* pCond = conditionals[n][i];
			for (TUInt j = 0; j < N; j++)
			{
				const DetectedEdge *pNextE = fixedEdgeCandidates[n+1][j];
				double interEdgeDistance = (pNextE->location - pE->location) / x;
				if (interEdgeDistance <= 0)
					energy = 1E6;
				else
				{
					double distanceFromExpected = abs(expectedInterEdgeDistance - interEdgeDistance);
					energy = coeffConditional * distanceFromExpected * distanceFromExpected;
				}
				pCond[j] = energy;
			}
		}
	}
}

bool BarcodeDecoder::getFixedEdgeCandidates(const vector<DetectedEdge> &detectedEdges, vector<vector<const DetectedEdge*> > &fixedEdgeCandidates)
{
	static const int nPositiveEdges = symbology_->nTotalEdges() / 2, nNegativeEdges = symbology_->nTotalEdges() / 2;
	static const TUInt nFixedEdges = symbology_->nFixedEdges();
	const DetectedEdge *lastEdge = &(detectedEdges.back());
	int nDetectedPositiveEdges = (lastEdge->polarity == 1 ? lastEdge->nPreviousPositiveEdges + 1 : lastEdge->nPreviousPositiveEdges);
	int nDetectedNegativeEdges = (lastEdge->polarity == -1 ? lastEdge->nPreviousNegativeEdges + 1 : lastEdge->nPreviousNegativeEdges);
	int nRemainingNegativeEdges = nDetectedNegativeEdges - nNegativeEdges, nRemainingPositiveEdges = nDetectedPositiveEdges - nPositiveEdges;
	//Find which edges have the correct numbers of previous +ve and -ve edges, and are the right polarity.
	vector<vector<const DetectedEdge*> >::iterator pCandidates = fixedEdgeCandidates.begin();
	vector<DetectedEdge>::const_iterator pFirstCandidate = detectedEdges.begin();
	for (TUInt n = 0; n < nFixedEdges; n++)
	{
		const BarcodeSymbology::Edge *pEdge = symbology_->getFixedEdge(n);
		pCandidates->clear();
		int minNegEdges = pEdge->nPreviousNegativeEdges(), maxNegEdges = pEdge->nPreviousNegativeEdges() + nRemainingNegativeEdges;
		int minPosEdges = pEdge->nPreviousPositiveEdges(), maxPosEdges = pEdge->nPreviousPositiveEdges() + nRemainingPositiveEdges;
		while ( (pFirstCandidate->nPreviousNegativeEdges < minNegEdges) || (pFirstCandidate->nPreviousPositiveEdges < minPosEdges) )
			pFirstCandidate++;	//keep track of first possible edge candidate so as not to iterate from the beginning
		//Find which edges are possible candidates
		for (vector<DetectedEdge>::const_iterator pDetectedEdge = pFirstCandidate; pDetectedEdge != detectedEdges.end(); pDetectedEdge++)
		{
			//pDetectedEdge is guaranteed to have the min. # of edges, now see if it has less than the max numbers.
			if ( (pDetectedEdge->nPreviousNegativeEdges > maxNegEdges) || (pDetectedEdge->nPreviousPositiveEdges > maxPosEdges) )
				break;	//no further candidates past this TPointInt
			//pDetectedEdge at this point has the appropriate number of previous and consecutive edges, now we check its polarity
			if ( pDetectedEdge->polarity == pEdge->polarity() )
				pCandidates->push_back(&(*pDetectedEdge));
		}
		if (pCandidates->empty()) //no candidates found -> it is not possible to fit a barcode to the detected edges
			return false;
		pCandidates++;
	}
	return true;	//there is a possible fit
}

void BarcodeDecoder::getDigitEnergies(int dir, const vector<SymbolBoundary> &boundaries)
{
	//Convolve the barcode symbol with the patterns to get digit energies.
	vector<TUInt> pattern(6);
	//Calculate patterns and convolve
	bool isBackwards = (dir == BACKWARD);
	for (TUInt s = 0; s < nSymbols_; s++) //for all symbols
	{
		//Find the symbol boundaries and fundamental width
		const BarcodeSymbology::Symbol *pSym = symbology_->getDataSymbol(s);
		int symbolEdge = boundaries[s].leftEdge;
		double xSym = (boundaries[s].rightEdge - boundaries[s].leftEdge) / (double) pSym->width;
		int sgn = (pSym->bars.front()->isDark() ? 1 : -1);
		int sumConv = 0;
		int symbolIndex = (isBackwards ? nSymbols_ - 1 - s : s);
		for (TUInt d = 0; d < 10; d++)	//for all possible digits
		{
			//Get the digit pattern
			symbology_->getConvolutionPattern(d, xSym, isBackwards, pattern);
			//Convolve at the expected digit boundaries
			int *start = slice_.data() + ((int) (symbolEdge - pattern.front()));
			int conv = max(dotProduct(sgn, start, pattern), 1);
			sumConv += conv;
			convolutions_(d, symbolIndex) = conv;
		}
		for (TUInt d = 0; d < 10; d++)
		{
			energies_(d, symbolIndex) = -log(convolutions_(d, symbolIndex) / (double) sumConv);
		}
	}
}

int BarcodeDecoder::dotProduct(int sgn, const int *data, const vector<TUInt> &pattern)
{
	int width = pattern.back();
	int dataMean = (data[width] - *data) / width;
	vector<TUInt>::const_iterator j = pattern.begin();
	int patternSum = sgn * (*j);
	int c = sgn * ( data[*j] - *data );
	for (j++; j != pattern.end(); j++)
	{
		sgn *= -1;
		c += sgn * ( data[*j] - data[*(j-1)] );
		patternSum += sgn * (*j - *(j-1));
	}
	return (c - patternSum * dataMean) / width;
}

