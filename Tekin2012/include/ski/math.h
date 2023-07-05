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
 * Math header for general use
 * Ender Tekin
 */

#ifndef SKI_MATH_H_
#define SKI_MATH_H_

#include <cmath>
#include "ski/types.h"
#include "ski/type_traits.h"
#include <stdarg.h>
#undef max
#undef min
#undef MAXINT
#undef MININT
#undef MAXUINT
#include <limits>
#include "ski/log.h"

namespace ski
{

const double PI=(4*atan(1.0));
const double INF = std::numeric_limits<double>::infinity();
const double NEGINF = -INF;
static const int MAXINT = (std::numeric_limits<int>::max)();
static const int MININT = (std::numeric_limits<int>::min)();
static const int MAXUINT = (std::numeric_limits<TUInt>::max)();
static const int VERYLARGE = MAXINT/16; //more formal in future.

/**
 * returns the maximum number of a given type.
 * @return maximum value of a given type.
 */
template<typename T>
inline T MaxValue() {return (std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity() : (std::numeric_limits<T>::max)()); };

/**
 * returns the minimum number of a given type.
 * @return minimum value of a given type.
 */
template<typename T>
inline T MinValue() {return (std::numeric_limits<T>::has_infinity ? -std::numeric_limits<T>::infinity() : (std::numeric_limits<T>::min)() ); };

/**
 * Integer power overload
 * @param[in] aNumber integer to take power of.
 * @param[in] aPow power
 * @return aNumber ^ aPow.  Note that the number can easily overflow.  To avoid overflow, use a float or double version.
 */
inline int pow(int aNumber, int aPow) {return (aPow == 0 ? 1 : aNumber * pow(aNumber, aPow-1) ); };

/**
 * Round
 * @param[in] a value to round
 * @return value rounded towards infinity
 */
inline double round( double value ) {return floor( value + 0.5 ); };

template <bool shouldBeRounded>
struct cast_selector
{
	template <typename T1, typename T2>
	inline static T2 cast(T1 val) {return (shouldBeRounded ? floor(val + 0.5): val); };
};

template <typename T1, typename T2>
inline T2 cast(T1 val) {return cast_selector<should_be_rounded<T1,T2>::value>::cast(val);  };

template <typename T>
inline T cast (T val) { return val; };

///Template class to use the correct method.
template <bool isImplemented>
struct method_selector
{
	template <class T>
	inline static double norm(const T& obj) {return std::abs(obj); };

	template <class T>
	inline static double distance(const T& obj1, const T& obj2) {return norm(obj1-obj2); };
};

//Uses the object's norm method if the object is designated as having implemented norm()
template <>
struct method_selector<true>
{
	template <class T>
	inline static double norm(const T& obj) {return obj.norm(); };

	template <class T>
	inline static double distance(const T& obj1, const T& obj2) {return obj1.distance(obj2); };
};

template <typename T>
inline double norm(T val) {return std::abs((double) val); };

/*
template <class T>
inline double norm(const T &a)
{
	return method_selector<implements<T>::norm>::norm(a);
};
*/

/**
 * Distance between two objects
 * @param[in] a object 1.
 * @param[in] b object 2.
 * If T is a complex class, the objects must implement a.norm(), b.norm() and the difference operator.
 * @return distance between a and b.
 */
template <class T>
inline double distance(const T &a, const T &b) {return norm(a-b); };
};	//end namespace ski

#endif // SKI_MATH_H_
