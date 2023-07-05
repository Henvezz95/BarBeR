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
 * @file Barcode Location and Decoding Engine
 * @author Ender Tekin
 */

#ifndef BLADE_H_
#define BLADE_H_

#include "ski/types.h"
#include <memory>
#include "ski/BLaDE/Barcode.h"

//Forward declarations
class BarcodeSymbology;
class _BLaDE;

/**
 * @class Barcode Location and Decoding Engine High-Level Access
 */
class BLaDE
{
public:
	/**
	 * BLaDE default options
	 */
	struct Options
	{
		/** Scale used for the finder */
		TUInt scale;
		/** Minimum number of cells a barcode needs to contain.*/
		TUInt nOrientations;
		/**
		 * Constructor
		 * @param[in] s scale to work at
		 * @param[in] n how finely to quantize orientation search
		 */
		Options(TUInt s=0, TUInt n=18):
			scale(s),
			nOrientations(n)
		{};
	};

	/**
	 * BLaDE included symbologies
	 */
	enum PredefinedSymbology
	{
		UPCA = 1
	};

	/**
	 * Constructor
	 * @param[in] aImg input image to work on
	 * @param[in] opts options to use
	 */
	BLaDE(const TMatrixUInt8 &aImg, const Options &opts=Options());

	/**
	 * Destructor
	 */
	~BLaDE();

	/**
	 * Returns a list of located barcodes on the image associated with this engine
	 * @return a list of detected possible barcodes
	 */
	BarcodeList& locate();

	/**
	 * Add symbology to use for decoding. Symbologies are tried in the order they are added.
	 * @param[in] aSymbology a symbology to try when attempting to decode
	 */
	void addSymbology(BarcodeSymbology* aSymbology);

	/**
	 * Adds a pre-defined symbology (with default options) to use for decoding.
	 * Symbologies are tried in the order they are added.
	 * @param[in] aSymbology a symbology to try when attempting to decode
	 */
	void addSymbology(PredefinedSymbology aSymbology);

	/**
	 * Attempt to decode a barcode
	 * @param[in, out] bc a located barcode returned by getLocator->locate()
	 * @return true if one of the symbologies has correctly decoded the barcode
	 */
	bool decode(Barcode &bc);

private:
	std::unique_ptr<_BLaDE> blade_;
};

#endif
