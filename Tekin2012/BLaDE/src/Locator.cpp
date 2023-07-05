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
 * @file
 * Barcode Location Code.
 * @author Ender Tekin
 */

#include "Locator.h"
#include "ski/math.h"
#include "ski/log.h"
#include <assert.h>

BarcodeLocator::BarcodeLocator(const TMatrixUInt8 &img, const Options &opts/* Options()*/):
	opts_(opts),
	image_(img, opts),
	mapPixelToCell_(image_.size())
{
	prepareCells();
	prepareTrigLookups();
	prepareScanLines();
}

BarcodeLocator::~BarcodeLocator()
{
}

void BarcodeLocator::locate(BarcodeList& barcodes)
{
	//Initialize list of barcodes
	barcodeCandidates_.clear();
	barcodes.clear();
	try
	{
		vector<Vote> orientationModes;
		//Get votes from pixels with gradients above threshold
		getOrientationCandidates(orientationModes);
		//Tally the resulting votes to estimate barcode orientation
		getBarcodeCandidates(orientationModes);
		//If barcodes found, sort barcodes
		LOGD("%u barcode candidates found\n", barcodeCandidates_.size());
		if (barcodeCandidates_.size())
		{
			//Sort the barcodes
			barcodeCandidates_.sort();
			//Reverse the order, so that bigger barcodes are first
			barcodeCandidates_.reverse();
			//Return the results
			for (BarcodeCandidateList::const_iterator pCandidate = barcodeCandidates_.begin(); pCandidate != barcodeCandidates_.end(); pCandidate++)
				barcodes.push_back(pCandidate->promote(opts_.scale));
		}
	}
	catch (exception &aErr)
	{
		LOGE("Locator error\n");
		throw;	//TODO: see if we throw without logging.
	}
}

void BarcodeLocator::getOrientationCandidates(vector<Vote> &orientationModes)
{
	//Calculate gradients
	image_.update();
	//Calculate histograms for each cell
	calculateCellHistograms();
	//Calculate votes for the orientation histogram
	calculateOrientationHistogram();
	//Find modes of the orientation histogram
	findOrientationHistogramModes(orientationModes);
}

void BarcodeLocator::calculateCellHistograms()
{
	static const TUInt M = image_.size().height, N = image_.size().width;
	//initialize histograms and voters
	for (vector<vector<Cell> >::iterator pCellRow = cells_.begin(); pCellRow != cells_.end(); pCellRow++)
	{
		for (vector<Cell>::iterator pCell = pCellRow->begin(); pCell != pCellRow->end(); pCell++)
			pCell->reset();
	}
	//TODO: use matrix class with iterator
	//Scan points and populate the histograms of corresponding cell
	static const TMatrixUInt8& magnitude = image_.magnitudes(), orientation = image_.orientations();
	for (TUInt i = 0; i < M; i++)
	{
		const TUInt8 *magRowPtr = magnitude[i], *magRowEnd = magnitude[i] + N, *angRowPtr = orientation[i];
		for (Cell **mapRowPtr = mapPixelToCell_[i]; magRowPtr < magRowEnd; magRowPtr++, angRowPtr++, mapRowPtr++)
		{
			if ( *magRowPtr )
				(*mapRowPtr)->addVoter(*angRowPtr, *magRowPtr);
		}
	}
}

void BarcodeLocator::calculateOrientationHistogram()
{
	//Calculate votes for overall histogram
	orientationHistogram_.assign(2 * opts_.nOrientations, 0); //initialize to 0.
	vector<TUInt>::iterator hCell, h;
	for (vector<vector<Cell> >::iterator pCellRow = cells_.begin(); pCellRow != cells_.end(); pCellRow++)
	{
		for (vector<Cell>::iterator pCell = pCellRow->begin(); pCell != pCellRow->end(); pCell++)
		{
			//If enough pixels are active and entropy is low, this cell will be considered for voting
			if ( pCell->shouldBeConsidered() )
			{
				//pCell->isSet = true;
				//add the cell histogram (unweighted) to the global histogram.
				for (TUInt o = 0; o < 2 * opts_.nOrientations; o++)
					orientationHistogram_[o] += pCell->orientationHistogram[o];
			}
		}
	}
}

void BarcodeLocator::findOrientationHistogramModes(vector<Vote> &orientationModes)
{
	vector<Vote> orientationVotes, orientationShiftedVotes;
	//mean shift over orientations
	for (TUInt o = 0; o < opts_.nOrientations; o++)
	{
		TUInt m = min(orientationHistogram_[o], orientationHistogram_[o + opts_.nOrientations]);
		if (m > opts_.minVotesPerOrientation)
			orientationVotes.push_back( Vote(o, m) );
	}
	//steepest ascent from each active orientation to find modes of distribution
	ascendModes(orientationVotes, orientationShiftedVotes);
	//Now find the modes - many have hopefully converged to a few
	static const double maxInterModeDistance = .5;
	findClusterCenters(orientationShiftedVotes, orientationModes, maxInterModeDistance);
}

void BarcodeLocator::ascendModes(const vector<Vote> &votes, vector<Vote> &modes)
{
	static const double tolerance = 0.0001, alpha = 0.1, beta = 0.5, var = 4;
	TUInt nVotes = votes.size();
	modes.assign(votes.begin(), votes.end());
	if (nVotes == 0)
		return;
	vector<Vote> weightedVotes(votes.begin(), votes.end());
	GaussianKernelD kernel(var);
	GaussianKernelRot kernel2(var, .5 * opts_.nOrientations);
	for (vector<Vote>::iterator v = modes.begin(); v != modes.end(); v++)
	{
		double grad, step, dist;
		do	//until convergence
		{
			//calculate f(theta^t)
			v->weight = kde(votes, v->loc, kernel2);
			//calculate gradient of f(theta) at theta^t. step = stepSize * gradient
			for (TUInt i = 0; i < nVotes; i++)
			{
				dist = votes[i].loc - v->loc;
				if (dist > opts_.nOrientations/2)
					dist -= opts_.nOrientations;
				else if (dist < -opts_.nOrientations/2)
					dist += opts_.nOrientations;
				weightedVotes[i].weight = votes[i].weight * dist / var;
				weightedVotes[i].loc = dist;
			}
			//gradient of kde
			step = grad = kde(weightedVotes, (double) 0, kernel);
			if (step > 1)	//limit max step size
				step = 1;
			//line search for best step size
			while ( kde(votes, v->loc + step, kernel2) < v->weight + alpha * step * grad )
				step *= beta;
			//f(theta^t+1) = f(theta) + step
			v->loc += step;
			//keep v->loc between 0 and opts_.nOrientations
			if (v->loc < 0)
				v->loc += opts_.nOrientations;
			else if (v->loc >= opts_.nOrientations)
				v->loc -= opts_.nOrientations;
		}
		while (abs(step) > tolerance);
	}
}

void BarcodeLocator::getBarcodeCandidates(const vector<Vote> &modes)
{
	//Filter out modes that have few votes
	for (vector<Vote>::const_iterator m = modes.begin(); m != modes.end(); m++)
	{
		//Find the dominant orientation horizontally and vertically, and see where we expect the barcode to lie.
		vector<TPointInt> candidates; //vector of candidate centers for barcode at this orientation
		getCandidateCellClusters(m->loc, candidates);
		//Now scan lines through the barcode area and see if this does look like a barcode
		TUInt8 orientation = floor(m->loc);
		if (m->loc >= (orientation + .5))
			orientation = (orientation + 1) % opts_.nOrientations;
		for (vector<TPointInt>::const_iterator p = candidates.begin(); p != candidates.end(); p++)
		{
			BarcodeCandidate aBC(orientation);	//barcode candidate - will be saved if passes the scan
			if ( scanSegment(aBC, *p) )
				barcodeCandidates_.push_back(aBC); //verified barcode candidate found, save
		}
	}
}

void BarcodeLocator::getCandidateCellClusters(double theta, vector<TPointInt> &candidates)
{
	//use the floor and ceiling of the mode to find barcode candidate limits.
	TUInt8 thetaQuantFloor = (TUInt8) floor(theta), thetaQuantCeil = ( (thetaQuantFloor + 1) % opts_.nOrientations );
	static GaussianKernelPt kernel(5 * opts_.cellSize);
	vector<VoteP> votes, shiftedVotes, clusterCenters;
	for (vector<vector<Cell> >::iterator pRow = cells_.begin(); pRow != cells_.end(); pRow++)
	{
		for (vector<Cell>::iterator pCell = pRow->begin(); pCell != pRow->end(); pCell++)
		{
			if ( (pCell->shouldBeConsidered()) &&
					( (pCell->dominantOrientation() == thetaQuantFloor) || (pCell->dominantOrientation() == thetaQuantCeil) ) )
				votes.push_back( VoteP( pCell->center(), (double) pCell->nVoters() ) );
		}
	}
	//mean shift to find cluster centers
	meanShift(votes, shiftedVotes, kernel);
	findClusterCenters(shiftedVotes, clusterCenters, 5);
	candidates.clear();
	for (vector<VoteP>::iterator c = clusterCenters.begin(); c != clusterCenters.end(); c++)
		candidates.push_back(c->loc);
}

bool BarcodeLocator::scanSegment(BarcodeCandidate &aBC, const TPointInt &pt)
{
	bool *isAcceptable = isAcceptable_[aBC.orientation];
	double theta = (ski::PI / opts_.nOrientations)* aBC.orientation;
	TPointDouble step(cos(theta), sin(theta));
	static const TRectInt imageRect(TPointInt(0,0), image_.size());
	if (!imageRect.contains(pt))
		LOGE("Why is this being called with a cluster center outside the image rectangle?\n");
	const TMatrixUInt8 magnitude = image_.magnitudes(), orientation = image_.orientations();
	aBC.nEdges = 0;
	for (int dir = 0; dir < 2; dir++) //starting from a TPointInt in the middle, extend in both directions to find the extend
	{
		int dist = 0;	//starting new trace
		TPointDouble curPt = pt;
		TPointInt lastEdge;
		if (dir == 1)
			step *= -1.0;
		while (imageRect.contains(curPt)) //TODO: ensure that pt is contained in imageRect and avoid check
		{
			curPt += step;
			if (magnitude(curPt))
			{
				if (isAcceptable[orientation(curPt)])	//correctly oriented edge - increase count and reset distance
				{
					lastEdge = curPt;
					dist = 0;
					aBC.nEdges++;
				}
				else if (aBC.nEdges > 0)	//unacceptable edge during trace - increase distance and decrease count
				{
					dist++;
					aBC.nEdges--;
				}
			}
			else if (aBC.nEdges > 0)
				dist++;	//no edge but tracing, just increase distance
			if (dist > opts_.maxDistBtwEdges) //if no correctly oriented TPointInt seen in a while, end trace
			{
				if (dir == 0)
					aBC.lastEdge = lastEdge;
				else
					aBC.firstEdge = lastEdge;
				break;
			}
		} //switch direction
	}
	//See if the "edge density" is above the threshold, and save if it is.
	LOGD("Barcode detected at (%d,%d) and orientation %d has %d edges\n", pt.x, pt.y, aBC.orientation, aBC.nEdges);
	//TODO: check the following line!!
	return ( aBC.nEdges > std::max( opts_.minEdgesInBarcode, (int) (aBC.width() * opts_.minEdgeDensityInBarcode) ) );
}

void BarcodeLocator::prepareCells()
{
	//Prepare cell vector and pixel->cell address lookup matrix when first called, reuse later
	TSizeInt cellSize(opts_.cellSize, opts_.cellSize);
	const TUInt M = image_.size().height, N = image_.size().width;
	TUInt MCell = M / cellSize.height, NCell = N / cellSize.width; //# of rows and columns of "cells"
	if (cellSize.height * MCell < M)
		MCell++;
	if (cellSize.width * NCell < N)
		NCell++;
	//populate matrix of cells
	Cell aCell(TRectInt(TPointInt(), cellSize), opts_.nOrientations, opts_.maxEntropy);
	vector<Cell> aCellRow;
	for (TUInt iCell = 0; iCell < MCell; iCell++)
	{
		aCellRow.clear();
		aCell.box.x = 0;
		aCell.box.height = (iCell < MCell-1 ? cellSize.height : M - aCell.box.y);
		for (TUInt jCell = 0; jCell < NCell; jCell++)
		{
			aCell.box.width = (jCell < NCell-1 ? cellSize.width: N - aCell.box.x);
			aCellRow.push_back(aCell);
			aCell.box.x += cellSize.width;
		}
		cells_.push_back(aCellRow);
		aCell.box.y += cellSize.height;
	}
	//create the pixel->cell map
	for (TUInt i = 0; i < M; i++)
	{
		Cell **mapRow = mapPixelToCell_[i];
		TUInt index = i / cellSize.height;
		mapRow[0] = &(cells_[index][0]);
		for (TUInt j = 0; j < (TUInt) N; j++)
			mapRow[j] = &(cells_[index][j / cellSize.width]);
	}
}

void BarcodeLocator::prepareTrigLookups()
{
	orientationHistogram_.assign(2*opts_.nOrientations, 0);
	TUInt M = image_.size().height, N = image_.size().width;
	int maxDim = max(M,N);
	//Trigonometric lookup tables
	cosLookupTable_ = TMatrixInt(opts_.nOrientations, maxDim);
	sinLookupTable_ = TMatrixInt(opts_.nOrientations, maxDim);
	for (TUInt o = 0; o < opts_.nOrientations; o++)
	{
		int *cosAtO = cosLookupTable_[o], *sinAtO = sinLookupTable_[o];
		double theta = o * ski::PI / opts_.nOrientations, sinTheta = sin(theta), cosTheta = cos(theta);
		for (int i = 0; i < maxDim; i++)
		{
			sinAtO[i] = (int) (i * sinTheta); //always positive for 0 <= theta < pi
			cosAtO[i] = (int) (i * cosTheta); //may be negative
		}
	}
}

void BarcodeLocator::prepareScanLines()
{
	vector<TPointInt> scanline;
	TUInt maxDim = max(image_.size().height, image_.size().width);
	isAcceptable_ = TMatrixBool(opts_.nOrientations, opts_.nOrientations * 2);
	TPointInt p, q;
	for (TUInt o = 0; o < opts_.nOrientations; o++)
	{
		p = TPointInt(0,0);
		//fill scan lines
		scanline.assign(1, p);
		int *cosTheta = cosLookupTable_[o], *sinTheta = sinLookupTable_[o];
		for (TUInt i = 1; i < maxDim; i++)
		{
			q = TPointInt(cosTheta[i], sinTheta[i]);
			scanline.push_back( q - p ); //store offsets
			p = q;
		}
		scanLines_.push_back(scanline);
		// fill acceptable orientations - isAcceptable(i,j) is true iff orientation j is acceptable for a barcode at orientation i
		static const int ALLOWED_DIST = 2;
		for (int n = 0; n < (int) opts_.nOrientations; n++)
		{
			for (int m = 0; m < (int) opts_.nOrientations; m++)
			{
				if ( ( abs(n - m) <= ALLOWED_DIST) || ( abs(n - m) >= (opts_.nOrientations - ALLOWED_DIST) ) )
					isAcceptable_(n, m) = isAcceptable_(n, m + opts_.nOrientations) = true;
				else
					isAcceptable_(n, m) = isAcceptable_(n, m + opts_.nOrientations) = false;
			}
		}
	}
}

//==============================
//
// IMAGECONTAINER
//
//==============================

BarcodeLocator::ImageContainer::ImageContainer(const TMatrixUInt8 &img, const BarcodeLocator::Options &opts):
		original(img),
		scale(opts.scale),
		outputSize(img.size().width >> scale, img.size().height >> scale),
		scaled(opts.scale > 0 ? TMatrixUInt8(outputSize) : TMatrixUInt8(0,0)),
		dI(outputSize),
		dJ(outputSize),
		dMag(outputSize),
		dAng(outputSize),
		tmp1(outputSize.width, outputSize.height),	//temporary areas are transposed
		tmp2(outputSize.width, outputSize.height),
		gradientOrientationMap(MAX_GRAD - MIN_GRAD + 1, MAX_GRAD - MIN_GRAD + 1),
		gradientMagnitudeMap(MAX_GRAD - MIN_GRAD + 1, MAX_GRAD - MIN_GRAD + 1)
{
	prepareGradientCalculator(opts.gradThresh, 2 * opts.nOrientations);
}

void BarcodeLocator::ImageContainer::prepareGradientCalculator(TUInt8 thresh, TUInt8 nOrientations)
{
    //Calculate magnitude and orientation maps used for mapping i/j gradient values to magnitude/quantized angle values
	int diNorm, djNorm;
	double angle, dTheta = 2 * PI / nOrientations;
	TUInt mag, thresh2 = (TUInt) thresh * (TUInt) thresh;
	for (int di = MIN_GRAD; di <= MAX_GRAD; di++)
	{
		diNorm = di - MIN_GRAD;
		for (int dj = MIN_GRAD; dj <= MAX_GRAD; dj++)
		{
			djNorm = dj - MIN_GRAD;
			mag = (TUInt) (di*di + dj*dj);
			gradientMagnitudeMap(diNorm, djNorm) = (TUInt8) (mag > thresh2 ? sqrt((double) (mag>>1)) : 0);	//scaled to ensure it will fit in TUInt8.
			if (gradientMagnitudeMap(diNorm, djNorm))
			{
				angle = atan2((double) di, (double) dj);
				gradientOrientationMap(diNorm, djNorm) = ((TUInt8) (angle / dTheta + 0.5 + nOrientations)) % nOrientations;
			}
			else
				gradientOrientationMap(diNorm, djNorm) = nOrientations;
		}
	}
}

void BarcodeLocator::ImageContainer::update()
{
	if (isSubsampled())
	{
		subsample(original, scaled, scale);
		calculateGradients(scaled);
	}
	else
		calculateGradients(original);
}

void BarcodeLocator::ImageContainer::subsample(const TMatrixUInt8 &input, TMatrixUInt8 &output, TUInt scale)
{
	assert(scale > 0);	//should only be used if scale is nonzero
	assert( (output.rows == (input.rows >> scale)) && (output.cols == (input.cols >> scale)) );	//make sure output is correct size
	TUInt M = input.rows, N = input.cols;
	LOGD("Subsampling image from %dx%d to %dx%d\n", M, N, output.rows, output.cols);
	//Subsample image
	TUInt step = 1 << scale;
	const TUInt8 *data;
	TUInt8 *dataScaled;
	for (TUInt i = 0, ii = 0; i < M; i+= step, ii++) //TODO: speed up this step
	{
		data = input[i];
		dataScaled = output[ii];
		for (TUInt j = 0, jj = 0; j < N; j += step, jj++)
			dataScaled[jj] = data[j];
	}
}

void BarcodeLocator::ImageContainer::calculateGradients(const TMatrixUInt8& input)
{
	//Calculate i/j gradients using separable Scharr operator
	calculateScharrGradients(input, dI, dJ, tmp1, tmp2);
	//Convert i/j gradients to polar gradients
	calculatePolarGradients(dI, dJ, dMag, dAng, gradientMagnitudeMap, gradientOrientationMap);
}

void BarcodeLocator::ImageContainer::calculateScharrGradients(const TMatrixUInt8 &img,
		TMatrixInt &iGrad, TMatrixInt &jGrad, TMatrixInt &tmp1, TMatrixInt &tmp2)
{
	//make sure matrices are correct size
	assert( (iGrad.rows == img.rows) && (iGrad.cols == img.cols) );
	assert( (jGrad.rows == img.rows) && (jGrad.cols == img.cols) );
	assert( (tmp1.rows == img.cols) && (tmp1.cols == img.rows) );
	assert( (tmp2.rows == img.cols) && (tmp2.cols == img.rows) );

	TUInt M = img.size().height, N = img.size().width;
    const TUInt pyStep = N, pyStepT = M; //y step for the transposed matrix
	//----------------
	//BORDERS
	//----------------
	TInt *iGradData, *iGradData2, *jGradData, *jGradData2, *tmp1Data, *tmp1Data2, *tmp2Data, *tmp2Data2;
	//rows M1 and M2 (corresponding to columns M1 and M2 in the tmp matrices)
	iGradData = iGrad[0]; iGradData2 = iGrad[M-1];
	jGradData = jGrad[0]; jGradData2 = jGrad[M-1];
	tmp1Data = tmp1[0]; tmp1Data2 = tmp1[0] + M-1;
	tmp2Data = tmp2[0]; tmp2Data2 = tmp2[0] + M-1;
	for (TInt* iGradDataFinal = iGrad[0]+N-1; iGradData <= iGradDataFinal;
			iGradData++, jGradData++, iGradData2++, jGradData2++,
					tmp1Data += pyStepT, tmp1Data2 += pyStepT, tmp2Data += pyStepT, tmp2Data2 += pyStepT)
		*iGradData = *jGradData = *iGradData2 = *jGradData2 = *tmp1Data = *tmp1Data2 = *tmp2Data = *tmp2Data2 = 0;
	//columns N1 and N2 (corresponding to rows N1 and N2 in the tmp matrices)
	iGradData = iGrad[0]; iGradData2 = iGrad[0] + N-1;
	jGradData = jGrad[0]; jGradData2 = jGrad[0] + N-1;
	tmp1Data = tmp1[0]; tmp1Data2 = tmp1[N-1];
	tmp2Data = tmp2[0]; tmp2Data2 = tmp2[N-1];
	for (TInt* iGradDataFinal = iGrad[M-1]; iGradData <= iGradDataFinal;
			iGradData += pyStep, jGradData += pyStep, iGradData2 += pyStep, jGradData2 += pyStep,
					tmp1Data++, tmp1Data2++, tmp2Data++, tmp2Data2++)
		*iGradData = *jGradData = *iGradData2 = *jGradData2 = *tmp1Data = *tmp1Data2 = *tmp2Data = *tmp2Data2 = 0;
	//----------------
	// INTERIOR
	//----------------
	//HORIZONTAL - we store the results in a transposed form to speed up the next stage (i.e. improve caching)
	const TUInt8 *imgRowBegin = img[0], *imgRowEnd = img[0] + N - 3;
	TInt *tmp1ColBegin = tmp1[1], *tmp2ColBegin = tmp2[1];
	//VERTICAL - becomes horizontal on transposed tmp matrices (improved caching)
	TInt *tmp1RowBegin = tmp1[0], *tmp1RowEnd = tmp1[0] + M - 3;
	TInt *tmp2RowBegin = tmp2[0], *tmp2RowEnd = tmp2[0] + M - 3;
	TInt *iGradColBegin = iGrad[0] + 1, *jGradColBegin = jGrad[0] + 1;
	//for each row of image -> column of tmp
	for (const TUInt8* imgRowBeginFinal = img[M-1]; imgRowBegin <= imgRowBeginFinal;
			imgRowBegin += pyStep, imgRowEnd += pyStep, tmp1ColBegin++, tmp2ColBegin++)
	{
		tmp1Data = tmp1ColBegin;
		tmp2Data = tmp2ColBegin;
		//for each column of image -> row of tmp
		for (const TUInt8* imgData = imgRowBegin; imgData <= imgRowEnd;
				imgData++, tmp1Data += pyStepT, tmp2Data += pyStepT)
		{
			*tmp1Data = ((TInt) *imgData) - ((TInt) imgData[2]);
			*tmp2Data = 3 * ((TInt) *imgData) + 10 * ((TInt) imgData[1]) + 3 * ((TInt) imgData[2]);
		}
	}
	//VERTICAL - becomes horizontal on transposed tmp matrices (improved caching)
	//for each row N1..N2 of tmp -> column of grad
	for (TInt *tmp1RowBeginFinal = tmp1[N-1]; tmp1RowBegin <= tmp1RowBeginFinal;
			tmp1RowBegin += pyStepT, tmp1RowEnd += pyStepT, tmp2RowBegin += pyStepT, tmp2RowEnd += pyStepT,
					iGradColBegin++, jGradColBegin++)
	{
		iGradData = iGradColBegin;
		jGradData = jGradColBegin;
		//for each column M1..M2 of this tmp row -> row of this grad column
		for (tmp1Data = tmp1RowBegin; tmp1Data <= tmp1RowEnd; tmp1Data++, jGradData += pyStep)
			*jGradData = (3 * tmp1Data[0] + 10 * tmp1Data[1] + 3 * tmp1Data[2]) / 16;
		for (tmp2Data = tmp2RowBegin; tmp2Data <= tmp2RowEnd; tmp2Data++, iGradData += pyStep)
			*iGradData = (*tmp2Data - tmp2Data[2]) / 16;
	}
}

void BarcodeLocator::ImageContainer::calculatePolarGradients(const TMatrixInt &iGrad, const TMatrixInt &jGrad,
		TMatrixUInt8 &absGrad, TMatrixUInt8 &angGrad, const TMatrixUInt8 &magnitudeLookup, const TMatrixUInt8 &orientationLookup)
{
	const TUInt M = iGrad.size().height, N = iGrad.size().width;
	const TInt *iGradData, *jGradData;
	TUInt8 *absGradData, *angGradData;
	TInt curDI, curDJ;
	const TInt *iGradRowBegin = iGrad[0], *iGradRowEnd = iGradRowBegin + N-1;
	const TInt *jGradRowBegin = jGrad[0];
	TUInt8 *absGradRowBegin = absGrad[0], *angGradRowBegin = angGrad[0];
	for (TUInt i = 0; i < M-1;
			i++, iGradRowBegin += N, iGradRowEnd += N, jGradRowBegin += N, absGradRowBegin += N, angGradRowBegin += N)
	{
		for (iGradData = iGradRowBegin, jGradData = jGradRowBegin, absGradData = absGradRowBegin, angGradData = angGradRowBegin;
				iGradData < iGradRowEnd; iGradData++, jGradData++, absGradData++, angGradData++)
		{
			//look up the magnitude and orientation of gradient for these di/dj values
			curDI = *iGradData + 255;
			curDJ = *jGradData + 255;
			*absGradData = magnitudeLookup(curDI, curDJ);
			*angGradData = orientationLookup(curDI, curDJ);	//TODO: see if it is faster to look these up or calculate
		}
	}
}

//==============================
//
// CELL
//
//==============================

BarcodeLocator::Cell::Cell(const TRectInt &aBox, TUInt nOrientations, double maxEntropy) :
	box(aBox),
	orientationHistogram(2 * nOrientations, 0),
	weightedOrientationHistogram(nOrientations, 0),
	dominantOrientation_(-1),
	entropy_(-1),
	nVoters_(0),
	maxEntropy_(maxEntropy),
	nOrientations_(nOrientations)
{
}

void BarcodeLocator::Cell::reset()
{
	orientationHistogram.assign(2 * nOrientations_, 0);
	weightedOrientationHistogram.assign(nOrientations_, 0);
	dominantOrientation_ = -1;
	entropy_ = -1.0;
	nVoters_ = 0;
}

void BarcodeLocator::Cell::addVoter(TUInt8 orientation, TUInt8 magnitude)
{
	orientationHistogram[orientation]++;
	weightedOrientationHistogram[orientation % nOrientations_] += magnitude;
	nVoters_++;
}

int BarcodeLocator::Cell::dominantOrientation()
{
	if (dominantOrientation_ < 0) //calculate from scratch
	{
		dominantOrientation_ = 0;
		for (TUInt o = 1; o < orientationHistogram.size(); o++)
		{
			if (orientationHistogram[o] > orientationHistogram[dominantOrientation_])
				dominantOrientation_ = o;
		}
	}
	return dominantOrientation_;
}

double BarcodeLocator::Cell::entropy()
{
	if (entropy_ < 0) //not previously calculated => calculate
	{
		entropy_ = 0;
		double prob, tot = 0.0;
		for (vector<TUInt>::iterator h = weightedOrientationHistogram.begin(); h < weightedOrientationHistogram.end(); h++)
		{
			if (*h)
			{
				prob = (double) *h;
				entropy_ -= prob * (log(prob));
				tot += prob;
			}
		}
		entropy_ = (tot > 0 ? log(tot) + entropy_ / tot : 0);
	}
	return entropy_;
}
