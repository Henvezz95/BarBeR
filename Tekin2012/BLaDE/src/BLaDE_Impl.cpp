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
 * Main entry point
 * @author Ender Tekin
 */

#include "BLaDE_Impl.h"
#include "ski/log.h"
#include "Locator.h"
#include "Decoder.h"
#include <stdexcept>
//Predefined symbologies
#include "UPCASymbology.h"


_BLaDE::_BLaDE(const TMatrixUInt8 &aImg, const BLaDE::Options &opts/*=Options()*/):
		opts_(opts),
		img_(aImg)
{
	BarcodeLocator::Options locatorOpts;
	locatorOpts.scale = opts.scale;
	locatorOpts.nOrientations = opts.nOrientations;
	locator_ = LocatorPtr(new BarcodeLocator(aImg, locatorOpts));
}

_BLaDE::~_BLaDE()
{
}

BarcodeList& _BLaDE::locate()
{
	locator_->locate(detectedBarcodes_);
	return detectedBarcodes_;
}

void _BLaDE::addSymbology(BarcodeSymbology* aSymbology)
{
	//check to make sure that a decoder for this symbology is not already in the list
	for (std::list<DecoderPtr>::const_iterator pDecoder = decoders_.begin(); pDecoder != decoders_.end(); pDecoder++)
	{
		if ((*pDecoder)->symbology() == aSymbology->name())
			throw std::logic_error("A decoder for this symbology is already registered");
	}
	//No such decoder registered, create
	//decoders_.emplace_back(DecoderPtr(new BarcodeDecoder(img_, aSymbology)));
	decoders_.push_back(DecoderPtr(new BarcodeDecoder(img_, aSymbology)));
}

void _BLaDE::addSymbology(BLaDE::PredefinedSymbology aSymbology)
{
	try
	{
		switch (aSymbology)
		{
		case BLaDE::UPCA:
			addSymbology(new UpcaSymbology());
			break;
		default:
			LOGE("No symbology class implementation is available for symbology %d\n", aSymbology);
			throw std::logic_error("Predefined symbologies defined in PredefinedSymbology must be matched to a class implementation in this method");
		}
	}
	catch (std::logic_error &aErr)
	{
		LOGE("A decoder for this symbology is already registered\n");
		throw;
	}
}

bool _BLaDE::decode(Barcode &bc)
{
	//Try each decoder in turn until one of them successfully decodes the barcode
	for (std::list<DecoderPtr>::iterator pDecoder = decoders_.begin(); pDecoder != decoders_.end(); pDecoder++)
	{
		BarcodeDecoder::Result res= (*pDecoder)->read(bc);
		switch (res)
		{
		case BarcodeDecoder::CANNOT_DECODE:
			LOGD("Barcode is not resolved sufficiently well to attempt decoding for symbology %s\n", (*pDecoder)->symbology().c_str());
			break;
		case BarcodeDecoder::DECODING_FAILED:
			LOGD("Failed to decode barcode with symbology %s\n", (*pDecoder)->symbology().c_str());
			break;
		case BarcodeDecoder::DECODING_SUCCESSFUL:
			LOGD("Successfully decoded barcode as %s with symbology %s\n", bc.estimate.c_str(), (*pDecoder)->symbology().c_str());
			return true;
		default:
			LOGE("Unknown barcode decoding status - should be handled!");
			break;
		}
	}
	return false;
}
