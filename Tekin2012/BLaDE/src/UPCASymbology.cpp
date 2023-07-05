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
 * UPCASymbology.cpp
 *
 *  Created on: Jul 16, 2012
 *      Author: kamyon
 */

#include "UPCASymbology.h"
#include "ski/log.h"
#include "ski/viterbi.h"

const TUInt UpcaSymbology::digitPatterns_[10][SYMBOL_LENGTH_] =
{
		{3, 2, 1, 1},
		{2, 2, 2, 1},
		{2, 1, 2, 2},
		{1, 4, 1, 1},
		{1, 1, 3, 2},
		{1, 2, 3, 1},
		{1, 1, 1, 4},
		{1, 3, 1, 2},
		{1, 2, 1, 3},
		{3, 1, 1, 2}
};

UpcaSymbology::UpcaSymbology(const Options& opts/*=Options()*/) :
		BarcodeSymbology("UPC-A"),
		opts_(opts),
		stateDigitMapForOddSymbol_(10,10),
		stateDigitMapForEvenSymbol_(10,10)
{
	//Set up symbology
	static const TUInt endBand[] = {1, 1, 1};
	static const TUInt midBand[] = {1, 1, 1, 1, 1};
	addSymbol(3, 3, endBand);
	for (TUInt n = 0; n < 6; n++)
		addSymbol(7, 4);
	addSymbol(5, 5, midBand);
	for (TUInt n = 0; n < 6; n++)
		addSymbol(7, 4);
	addSymbol(3, 3, endBand);

	//Set up joint decoder
	for (TUInt prevState = 0; prevState < 10; prevState++)
	{
		TUInt *odd = stateDigitMapForOddSymbol_[prevState];
		TUInt *even = stateDigitMapForEvenSymbol_[prevState];
		for (TUInt digit = 0; digit < 10; digit++)
		{
			odd[(3 * digit + prevState) % 10] = digit;
			even[(digit + prevState) % 10] = digit;
		}
	}
	LOGD("UPCA Symbology created\n");
}

UpcaSymbology::~UpcaSymbology() {}

void UpcaSymbology::getConvolutionPattern(TUInt digit, double x, bool isFlipped, vector<TUInt> &pattern) const
{
	pattern.resize(SYMBOL_LENGTH_ + 2);
	const TUInt *dP = digitPatterns_[digit];
	if (isFlipped)
		dP += SYMBOL_LENGTH_ - 1;
	vector<TUInt>::iterator p = pattern.begin();
	//extend one X prior to symbol
	double width = x;
	*p = (TUInt) x;
	//digit pattern in symbol
	for (p++; p != pattern.end() - 1; p++)
	{
		width += (*dP) * x;
		*p = (TUInt) width;
		isFlipped ? dP-- : dP++;
	}
	//extend one X past symbol
	*p = (TUInt) (width + x);
}

inline TUInt UpcaSymbology::getDigitFromStates(TUInt prevState, TUInt curState, TUInt symbol) const
{
	return (symbol%2 == 0 ? stateDigitMapForOddSymbol_(prevState, curState) : stateDigitMapForEvenSymbol_(prevState, curState));
}

string UpcaSymbology::estimate(const TMatEnergy &energies) const
{
	//-----------
	//Joint Viterbi estimation
	//-----------
	string upcaStr;
	static vector<vector<TEnergy> > priors;
	static vector<TMatEnergy> conditionals;
	TUInt nSymbols = nDataSymbols();
	if (priors.size() == 0)
	{
		priors.assign(nSymbols, vector<TEnergy>(10, (TEnergy) 0));
		conditionals.clear();
		conditionals.reserve(nSymbols - 1);
		for (TUInt i = 0; i < nSymbols - 1; i++)
			conditionals.push_back(TMatEnergy(10, 10, (TEnergy) 0));
	}
	//single Energies
	vector<TEnergy> &prior = priors.front();
	for (TUInt curState = 0; curState < 10; curState++)
		prior[curState] = energies(getDigitFromStates(0, curState, 0), 0);
	//Conditionals
	for (TUInt t = 1; t < nSymbols; t++)
	{
		TMatEnergy cond = conditionals[t-1];
		for (TUInt prevState = 0; prevState < 10; prevState++)
		{
			TEnergy *c = cond[prevState];
			for (TUInt curState = 0; curState < 10; curState++)
				c[curState] = energies(getDigitFromStates(prevState, curState, t), t);
		}
	}

	vector<TUInt> upcaEstimate;
	try
	{
		static Viterbi<TEnergy> V(priors, conditionals, 2);
		V.solve(0);
		vector<int> &bestseq = V.solutions[0].sequence;
		int prevState, curState = 0;
		upcaEstimate.resize(nSymbols);	//TODO: change bc struct
		for (TUInt t = 0; t < nSymbols; t++)
		{
			prevState = curState;
			curState = bestseq[t];
			upcaEstimate[t] = getDigitFromStates(prevState, curState, t);
		}

		//-----------
		//Checks
		//-----------
		//Margin test
		double margin = (V.solutions[1].energy - V.solutions[0].energy) / ((double) V.solutions[0].energy);

		if (margin < opts_.minMargin)
		{
			LOGD("Barcode estimate %s failed margin test (%f < %f)\n", convertEstimateToString(upcaEstimate).c_str(), margin, opts_.minMargin);
			return upcaStr;
		}
		//Individual most likely digits
		int nDifferentDigits = 0;
		for (TUInt symbol = 0; symbol < nSymbols; symbol++)
		{
			TUInt estimatedDigit = upcaEstimate[symbol];
			for (TUInt digit = 1; digit < 10; digit++)
			{
				if ( (digit != estimatedDigit) && (energies(digit, symbol) < energies(estimatedDigit, symbol)) )
				{
					nDifferentDigits++;
					break;
				}
			}
			if (nDifferentDigits > 1)
			{
				LOGD("Barcode estimate %s failed parity constraint (more than 1 digit is not most likely)\n", convertEstimateToString(upcaEstimate).c_str());
				return upcaStr;
			}
		}
		//Passed both checks -- convert to string
		//for (vector<TUInt>::const_iterator iDigit = upcaEstimate.begin(); iDigit != upcaEstimate.end(); iDigit++)
		//	upcaStr += ('0' + (char) *iDigit);
		upcaStr = convertEstimateToString(upcaEstimate);
		LOGD("Estimated barcode %s with energy = %f, margin = %4.3f\n", upcaStr.c_str(), (double) V.solutions[0].energy, margin);
	}
	catch (exception &aErr)
	{
		throw aErr;
	}
	return upcaStr;
}
