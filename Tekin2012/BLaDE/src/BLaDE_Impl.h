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

#ifndef BLADE_IMPL_H_
#define BLADE_IMPL_H_

#include "ski/types.h"
#include <list>
#include <memory>
#include "ski/BLaDE/BLaDE.h"

#ifdef USE_OPENCV
#define USING_OPENCV
#endif

//Forward declarations
class BarcodeLocator;
class BarcodeDecoder;
class BarcodeSymbology;

/**
 * @class Barcode Location and Decoding Engine High-Level Access
 */
class _BLaDE
{
public:

	/**
	 * Constructor
	 * @param[in] aImg input image to work on
	 * @param[in] opts options to use
	 */
	_BLaDE(const TMatrixUInt8 &aImg, const BLaDE::Options &opts=BLaDE::Options());

	/**
	 * Destructor
	 */
	~_BLaDE();

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
	void addSymbology(BLaDE::PredefinedSymbology aSymbology);

	/**
	 * Attempt to decode a barcode
	 * @param[in, out] bc a located barcode returned by getLocator->locate()
	 * @return true if one of the symbologies has correctly decoded the barcode
	 */
	bool decode(Barcode &bc);

private:
	/** Options used by BLaDE */
	BLaDE::Options opts_;
	/** Smart pointer to barcode locator */
	typedef std::unique_ptr<BarcodeLocator> LocatorPtr;
	/** Smart pointer to barcode decoder */
	typedef std::unique_ptr<BarcodeDecoder> DecoderPtr;

	/** Image to work on */
	const TMatrixUInt8 &img_;

	///List of detected barcodes
	BarcodeList detectedBarcodes_;

	/** Locator */
	LocatorPtr locator_;

	/** List of registered decoders (1 for each symbology) */
	std::list<DecoderPtr> decoders_;
};

#endif //BLADE_IMPL_H_
