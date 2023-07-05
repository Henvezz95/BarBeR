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
 * UPCASymbology.h
 *
 *  Created on: Jul 16, 2012
 *      Author: kamyon
 */

#ifndef UPCASYMBOLOGY_H_
#define UPCASYMBOLOGY_H_

#include "ski/BLaDE/Symbology.h"
#include "ski/types.h"

class UpcaSymbology: public BarcodeSymbology
{
public:
	/**
	 * Decoding options
	 */
	struct Options
	{
		/** minimum barcode energy margin to pass verification */
		double minMargin;
		/** maximum barcode energy to pass verification */
		double maxEnergy;
		/** Constructor */
		Options():
			minMargin(0.02),
			maxEnergy(20) //TODO: remove? doesn't seem to be used currently
		{};
	};

	/**
	 * Constructor
	 */
	UpcaSymbology(const Options &opts = Options());

	/**
	 * Destructor
	 */
	virtual ~UpcaSymbology();

	/**
	 * Returns a convolution pattern to the decoder
	 * @param[in] digit digit to return the pattern for
	 * @param[in] fundamental width
	 * @param[in] whether the pattern is horizontally flipped
	 * @param[out] pattern pattern to use for convolution.
	 */
	virtual void getConvolutionPattern(TUInt digit, double x, bool isFlipped, vector<TUInt> &pattern) const;

	/**
	 * Estimates the barcode from the matrix of digit energies for each symbol
	 * @param[in] energies matrix of digit energies per symbol
	 * @return a string that is the barcode estimate, empty string if estimate fails verification
	 */
	string estimate(const TMatEnergy &energies) const;

protected:
	/** Decoding options */
	Options opts_;

	/** Mapping from auxiliary variables to actual digits*/
	TMatrixUInt stateDigitMapForOddSymbol_;

	/** Mapping from auxiliary variables to actual digits*/
	TMatrixUInt stateDigitMapForEvenSymbol_;

	/**
	 * Returns the digit that results in transition from one state to the other for a given symbol
	 * @param[in] prevState previous state
	 * @param[in] curState current state
	 * @param[in] symbol current symbol
	 */
	inline TUInt getDigitFromStates(TUInt prevState, TUInt curState, TUInt symbol) const;

	/** Number of bars in each symbol */
	static const TUInt SYMBOL_LENGTH_ = 4;

	/** UPC Digit patterns */
	static const TUInt digitPatterns_[10][SYMBOL_LENGTH_];
};


#endif /* UPCASYMBOLOGY_H_ */
