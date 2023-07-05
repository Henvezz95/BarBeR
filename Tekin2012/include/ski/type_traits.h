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

#ifndef SKI_TYPE_TRAITS_H_
#define SKI_TYPE_TRAITS_H_
#include "ski/types.h"
#include <limits>
namespace ski
{

//Type traits for promotion
template<typename T1, typename T2>
struct promote
{
	//typedef T1 firstArg;
	//typedef T2 secondArg;
	typedef T2 result;
};

template<> struct promote<TInt8, TInt8> 	{typedef TInt32 result; };
template<> struct promote<TInt8, TInt16> 	{typedef TInt32 result; };
template<> struct promote<TInt8, TInt32> 	{typedef TInt32 result; };
template<> struct promote<TInt8, TInt64> 	{typedef TInt64 result; };
template<> struct promote<TInt8, TUInt8> 	{typedef TUInt32 result; };
template<> struct promote<TInt8, TUInt16> 	{typedef TUInt32 result; };
template<> struct promote<TInt8, TUInt32> 	{typedef TUInt32 result; };
template<> struct promote<TInt8, TUInt64> 	{typedef TUInt64 result; };
template<> struct promote<TInt8, float> 	{typedef float result; };

template<> struct promote<TInt16, TInt8> 	{typedef TInt32 result; };
template<> struct promote<TInt16, TInt16> 	{typedef TInt32 result; };
template<> struct promote<TInt16, TInt32> 	{typedef TInt32 result; };
template<> struct promote<TInt16, TInt64> 	{typedef TInt64 result; };
template<> struct promote<TInt16, TUInt8> 	{typedef TUInt32 result; };
template<> struct promote<TInt16, TUInt16>{typedef TUInt32 result; };
template<> struct promote<TInt16, TUInt32>{typedef TUInt32 result; };
template<> struct promote<TInt16, TUInt64>{typedef TUInt64 result; };
template<> struct promote<TInt16, float> 	{typedef float result; };

template<> struct promote<TInt32, TInt8> 	{typedef TInt32 result; };
template<> struct promote<TInt32, TInt16> 	{typedef TInt32 result; };
template<> struct promote<TInt32, TInt32> 	{typedef TInt32 result; };
template<> struct promote<TInt32, TInt64> 	{typedef TInt64 result; };
template<> struct promote<TInt32, TUInt> 	{typedef TInt32 result; };
template<> struct promote<TInt32, TUInt8> 	{typedef TInt32 result; };
template<> struct promote<TInt32, TUInt16>	{typedef TInt32 result; };
template<> struct promote<TInt32, TUInt64>	{typedef TUInt64 result; };
template<> struct promote<TInt32, float> 	{typedef float result; };

template<> struct promote<TInt64, TInt8> 	{typedef TInt64 result; };
template<> struct promote<TInt64, TInt16> 	{typedef TInt64 result; };
template<> struct promote<TInt64, TInt32> 	{typedef TInt64 result; };
template<> struct promote<TInt64, TInt64> 	{typedef TInt64 result; };
template<> struct promote<TInt64, TUInt8> 	{typedef TUInt64 result; };
template<> struct promote<TInt64, TUInt16>{typedef TUInt64 result; };
template<> struct promote<TInt64, TUInt32>{typedef TUInt64 result; };
template<> struct promote<TInt64, TUInt64>{typedef TUInt64 result; };
template<> struct promote<TInt64, float> 	{typedef float result; };

template<> struct promote<TUInt8, TInt8> 	{typedef TUInt32 result; };
template<> struct promote<TUInt8, TInt16> 	{typedef TUInt32 result; };
template<> struct promote<TUInt8, TInt32> 	{typedef TUInt32 result; };
template<> struct promote<TUInt8, TInt64> 	{typedef TUInt64 result; };
template<> struct promote<TUInt8, TUInt8> 	{typedef TUInt32 result; };
template<> struct promote<TUInt8, TUInt16>{typedef TUInt32 result; };
template<> struct promote<TUInt8, TUInt32>{typedef TUInt32 result; };
template<> struct promote<TUInt8, TUInt64>{typedef TUInt64 result; };
template<> struct promote<TUInt8, float> 	{typedef float result; };

template<> struct promote<TUInt16, TInt8>	{typedef TUInt32 result; };
template<> struct promote<TUInt16, TInt16> {typedef TUInt32 result; };
template<> struct promote<TUInt16, TInt32> {typedef TUInt32 result; };
template<> struct promote<TUInt16, TInt64> {typedef TUInt64 result; };
template<> struct promote<TUInt16, TUInt8> {typedef TUInt32 result; };
template<> struct promote<TUInt16, TUInt16>{typedef TUInt32 result; };
template<> struct promote<TUInt16, TUInt32>{typedef TUInt32 result; };
template<> struct promote<TUInt16, TUInt64>{typedef TUInt64 result; };
template<> struct promote<TUInt16, float> {typedef float result; };

template<> struct promote<TUInt32, TInt8> 	{typedef TUInt32 result; };
template<> struct promote<TUInt32, TInt16>{typedef TUInt32 result; };
template<> struct promote<TUInt32, TInt32>{typedef TUInt32 result; };
template<> struct promote<TUInt32, TInt64>{typedef TUInt64 result; };
template<> struct promote<TUInt32, TUInt> 	{typedef TUInt32 result; };
template<> struct promote<TUInt32, TUInt8>{typedef TUInt32 result; };
template<> struct promote<TUInt32, TUInt16>{typedef TUInt32 result; };
template<> struct promote<TUInt32, TUInt64>{typedef TUInt64 result; };
template<> struct promote<TUInt32, float> {typedef float result; };

template<> struct promote<TUInt64, TInt8> 	{typedef TUInt64 result; };
template<> struct promote<TUInt64, TInt16>{typedef TUInt64 result; };
template<> struct promote<TUInt64, TInt32>{typedef TUInt64 result; };
template<> struct promote<TUInt64, TInt64>{typedef TUInt64 result; };
template<> struct promote<TUInt64, TUInt8>{typedef TUInt64 result; };
template<> struct promote<TUInt64, TUInt16>{typedef TUInt64 result; };
template<> struct promote<TUInt64, TUInt32>{typedef TUInt64 result; };
template<> struct promote<TUInt64, TUInt64>{typedef TUInt64 result; };
template<> struct promote<TUInt64, float> {typedef float result; };

template<> struct promote<float, TInt8> 	{typedef float result; };
template<> struct promote<float, TInt16>{typedef float result; };
template<> struct promote<float, TInt32>{typedef float result; };
template<> struct promote<float, TInt64>{typedef float result; };
template<> struct promote<float, TUInt8>{typedef float result; };
template<> struct promote<float, TUInt16>{typedef float result; };
template<> struct promote<float, TUInt32>{typedef float result; };
template<> struct promote<float, TUInt64>{typedef float result; };
template<> struct promote<float, float> {typedef float result; };

template<typename T> struct promote<T, double> {typedef double result; };
template<typename T> struct promote<double, T> {typedef double result; };

template<> struct promote<double, double> {typedef double result; };	//to resolve ambiguity

//Type traits for rounding
/**
 * @param T1 type of input
 * @param T2 type of output
 */
template<typename T1, typename T2>
struct should_be_rounded
{
	/**
	 * Is true if a cv_type of T1 should be rounded when cast to a cv_type of T2 at compile time.
	 */
	static const bool value = (!std::numeric_limits<T1>::is_integer) && (std::numeric_limits<T2>::is_integer);
};

//Type traits for some operations
template <class T>
struct implements
{
	///Whether the "double T::norm()" function is implemented
	static const bool norm = false;
};
//Must be specializations for classes that implement the norm() operator

}; //namespace ski
#endif //SKI_TYPE_TRAITS_H_
