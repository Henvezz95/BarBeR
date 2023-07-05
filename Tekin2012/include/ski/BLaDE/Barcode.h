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
 * @file Barcode.h
 *
 *  Created on: May 24, 2011
 *  @author: Ender Tekin
 */

#ifndef BARCODE_H_
#define BARCODE_H_

#include "ski/cv.hpp"
#include <string>
#include <list>

/**
 * Structure containing information about decoded barcode.
 */
struct Barcode
{
	/** first edge of this barcode segment */
	TPointInt firstEdge;

	/** last edge of this barcode segment */
	TPointInt lastEdge;

	/** Barcode estimate information */
	std::string estimate;

	/** Type of barcode (determined by the decoder that produces a valid estimate) */
	std::string symbology;

	/**
	 * Constructor
	 * @param[in] pt1 one end of the barcode strip
	 * @param[in] pt2 other end of the barcode strip
	 * @param[in] est estimated barcode string
	 * @param[in] sym symbology used for the estimate
	 */
	Barcode(const TPointInt &pt1, const TPointInt &pt2) :
		firstEdge(pt1),
		lastEdge(pt2)
	{};
};

typedef std::list<Barcode> BarcodeList;

#endif /* BARCODE_H_ */
