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
 * @file Simple type declarations.
 * @author Ender Tekin
 */

#ifndef SKI_TYPES_H_
#define SKI_TYPES_H_

#define __STDC_CONSTANT_MACROS
#include "stdint.h"

//Simple types
typedef unsigned int TUInt;
typedef int TInt;
typedef uint8_t TUInt8;
typedef int8_t TInt8;
typedef uint16_t TUInt16;
typedef int16_t TInt16;
typedef uint32_t TUInt32;
typedef int32_t TInt32;
typedef uint64_t TUInt64;
typedef int64_t TInt64;

#endif //SKI_TYPES_H_
