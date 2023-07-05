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
 * BarcodeStruct.cpp
 *
 *  Created on: Nov 5, 2011
 *      Author: kamyon
 */

#include "ski/BLaDE/Symbology.h"
#include <stdexcept>
#include "ski/log.h"

//=========================================================
//BarcodeSymbology::Symbol
//=========================================================
BarcodeSymbology::Symbol::Symbol(TUInt aWidth, int dataSymbolIndex, vector<Bar*> aBars):
	width(aWidth),
	index(dataSymbolIndex),
	bars(aBars)
{
}

//=========================================================
//BarcodeSymbology
//=========================================================

BarcodeSymbology::Symbol* BarcodeSymbology::addSymbol(TUInt width, TUInt nBars, const TUInt *pattern/*=NULL*/)
{
	int loc = (symbols_.size() > 0 ? edges_.back().location : 0);
	//Add the corresponding bars
	vector<Bar*> bars;
	bars.reserve(nBars);
	bool isDataSymbol = (pattern == NULL);
	if (isDataSymbol)
	{
		//Add all bars until the last
		for (TUInt iBar = 0; iBar < nBars - 1; iBar++)
			bars.push_back(addBar(-1));
		//Add last bar
		bars.push_back(addBar(loc + width));
	}
	else	//is special symbol
	{
		for (TUInt iBar = 0; iBar < nBars; iBar++)
			bars.push_back(addBar(loc += pattern[iBar]));
	}
	symbols_.push_back(Symbol(width, (isDataSymbol ? dataSymbols_.size() : -1), bars));
	Symbol *lastSymbol = &(symbols_.back());
	if (isDataSymbol)
		dataSymbols_.push_back(lastSymbol);
	return lastSymbol;
}

BarcodeSymbology::Bar* BarcodeSymbology::addBar(int rightEdgeLocation)
{
	Edge *left = (bars_.size() ? &(edges_.back()) : addEdge(0)); //add left edge of barcode if no previous edges exist.
	//Add the corresponding right edge
	Edge *right = addEdge(rightEdgeLocation);
	//Add the bar
	bars_.push_back(Bar(left, right));
	return &(bars_.back());
}

BarcodeSymbology::Edge* BarcodeSymbology::addEdge(int loc)
{
	edges_.push_back(Edge(edges_.size(), loc));
	Edge *lastEdge = &(edges_.back());
	if (loc != -1)
		fixedEdges_.push_back(lastEdge);
	return lastEdge;
}

const BarcodeSymbology::Edge* BarcodeSymbology::getFixedEdge(TUInt i) const
{
	if (i > fixedEdges_.size())
		throw invalid_argument("Requested fixed edge does not exist");
	return fixedEdges_[i];
}

const BarcodeSymbology::Symbol* BarcodeSymbology::getDataSymbol(TUInt i) const
{
	if (i > dataSymbols_.size())
		throw invalid_argument("Requested data symbol does not exist");
	return dataSymbols_[i];
}

void BarcodeSymbology::getConvolutionPattern(TUInt digit, double x, bool isFlipped, vector<TUInt> &pattern) const {}

string BarcodeSymbology::convertEstimateToString(const vector<TUInt> &estimate) const
{
	string estimateStr;
	for (vector<TUInt>::const_iterator iDigit = estimate.begin(); iDigit != estimate.end(); iDigit++)
		estimateStr += ('0' + (char) *iDigit);
	return estimateStr;

}
