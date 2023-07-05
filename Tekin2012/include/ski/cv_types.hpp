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

/* @file ski_cv.hpp
*  Write own types for Point_, Mat, Mat_, Rect_, Size_, etc. (to be implemented)
 *
 *  Created on: Aug 10, 2012
 *      Author: kamyon
 */

#ifndef SKI_CV_TYPES_H_
#define SKI_CV_TYPES_H_

#include "ski/types.h"
#include "ski/math.h"
#include "ski/log.h"
#include "ski/type_traits.h"
#include <vector>
#include <memory>
#include <assert.h>

namespace ski
{

/// Two-dimensional point template
template <typename T>
struct TPoint2
{
	/** abscissa of the point */
	T x;
	/** ordinate of the point */
	T y;

	/**
	 * Constructor
	 * @param x_ abscissa
	 * @param y_ ordinate
	 */
	template <typename T2, typename T3>
	explicit TPoint2(T2 x_, T3 y_) :
		x(cast(x_)),
		y(cast(y_))
	{};

	/**
	 * Constructor
	 */
	explicit TPoint2() :
		x(0),
		y(0)
	{};

	template <typename T2>
	explicit TPoint2(const TPoint2<T2> &aPt) :
		x(cast(aPt.x)),
		y(cast(aPt.y))
	{};

	//Arithmetic operators
	/**
	 * Addition
	 */
	template <typename T2>
	TPoint2<typename promote<T,T2>::result> operator+ (const TPoint2<T2> &aPt) const
	{
		return TPoint2<typename promote<T,T2>::result>(x+aPt.x, y+aPt.y);
	};

	/**
	 * Addition & assignment
	 */
	template <typename T2>
	TPoint2<T>& operator+= (const TPoint2<T2> &aPt)
	{
		x += aPt.x;
		y+= aPt.y;
		return *this;
	};

	/**
	 * Subtraction
	 */
	template <typename T2>
	TPoint2<typename promote<T,T2>::result> operator- (const TPoint2<T2> &aPt) const
	{
		return TPoint2<typename promote<T,T2>::result>(x-aPt.x, y-aPt.y);
	};

	/**
	 * subtraction & assignment
	 */
	template <typename T2>
	TPoint2<T>& operator-= (const TPoint2<T2> &aPt)
	{
		x -= aPt.x;
		y -= aPt.y;
		return *this;
	};

	/**
	 * Multiplication
	 */
	template <typename T2>
	TPoint2<typename promote<T,T2>::result> operator* (T2 c) const
	{
		return TPoint2<typename promote<T,T2>::result>(x * c, y * c);
	};

	/**
	 * Multiplication & assignment
	 */
	template <typename T2>
	TPoint2<T>& operator*= (T2 c)
	{
		x *= c;
		y *= c;
		return *this;
	};

	/**
	 * Multiplication & assignment
	 */
	template <typename T2>
	const TPoint2<T>& operator*= (T2 c) const
	{
		x *= c;
		y *= c;
		return *this;
	};

	/**
	 * Division
	 */
	template <typename T2>
	TPoint2<typename promote<T,T2>::result> operator/ (T2 c) const
	{
		return TPoint2<typename promote<T,T2>::result>(x / c, y / c);
	};

	/**
	 * Division & assignment
	 */
	template <typename T2>
	TPoint2<T>& operator/= (T2 c)
	{
		x /= c;
		y /= c;
		return *this;
	};

	/**
	 * Division & assignment
	 */
	template <typename T2>
	const TPoint2<T>& operator/= (T2 c) const
	{
		x /= c;
		y /= c;
		return *this;
	};

	//Typecast
	/**
	 * Typecasts with rounding
	 */
	template <typename T2>
	operator TPoint2<T2>() const
	{
		return TPoint2<T2>(x, y);
	};

	//Comparisons
	/**
	 * Equals
	 */
	inline bool operator== (const TPoint2<T> &aPt) const
	{
		return ( (x == aPt.x) && (y == aPt.y) );
	};

	/**
	 * Not equals
	 */
	inline bool operator!= (const TPoint2<T> &aPt) const
	{
		return ( (x != aPt.x) || (y != aPt.y) );
	};

	//Other operations
	/**
	 * Norm
	 */
	template <typename T2>
	inline double dot(const TPoint2<T2> &aPt) const
	{
		return std::sqrt( (double) x * (double) aPt.x + (double) y * (double) aPt.y );
	};
};

template<typename T>
struct implements<class TPoint2<T> >
{
	static const bool norm = true;
};

template<typename T>
inline double norm(const TPoint2<T> &aPt)
{
	return aPt.dot(aPt);
};

/// Two-dimensional size template
template <typename T>
struct TSize
{
	/** width of the rectangle */
	T width;
	/** height of the rectangle */
	T height;

	/**
	 * Constructor
	 * @param w width
	 * @param h height
	 */
	template <typename T2, typename T3>
	explicit TSize(T2 w, T3 h) :
		width(cast(w)),
		height(cast(h))
	{};

	/**
	 * Constructor
	 */
	explicit TSize() :
		width(0),
		height(0)
	{};

	template <typename T2>
	explicit TSize(const TSize<T2> &aSz) :
		width(cast(aSz.width)),
		height(cast(aSz.height))
	{};

	//Arithmetic operators
	/**
	 * Addition
	 */
	template <typename T2>
	TSize<typename promote<T,T2>::result> operator+ (const TSize<T2> &aSz) const
	{
		return TSize<typename promote<T,T2>::result>(width+aSz.width, height+aSz.height);
	};

	/**
	 * Addition & assignment
	 */
	template <typename T2>
	TSize<T>& operator+= (const TSize<T2> &aSz)
	{
		width += aSz.width;
		height+= aSz.height;
		return *this;
	};

	/**
	 * Subtraction
	 */
	template <typename T2>
	TSize<typename promote<T,T2>::result> operator- (const TSize<T2> &aSz) const
	{
		return TSize<typename promote<T,T2>::result>(width-aSz.width, height-aSz.height);
	};

	/**
	 * subtraction & assignment
	 */
	template <typename T2>
	TSize<T>& operator-= (const TSize<T2> &aSz)
	{
		width -= aSz.width;
		height -= aSz.height;
		return *this;
	};

	/**
	 * Multiplication
	 */
	template <typename T2>
	TSize<typename promote<T,T2>::result> operator* (T2 c) const
	{
		return TSize<typename promote<T,T2>::result>(width * c, height * c);
	};

	/**
	 * Multiplication & assignment
	 */
	template <typename T2>
	TSize<T>& operator*= (T2 c) const
	{
		width *= c;
		height *= c;
		return *this;
	};

	/**
	 * Division
	 */
	template <typename T2>
	TSize<typename promote<T,T2>::result> operator/ (T2 c) const
	{
		return TSize<typename promote<T,T2>::result>(width / c, height / c);
	};

	/**
	 * Division & assignment
	 */
	template <typename T2>
	TSize<T>& operator/= (T2 c) const
	{
		width /= c;
		height /= c;
		return *this;
	};

	//Typecast
	/**
	 * Typecasts with rounding
	 */
	template <typename T2>
	operator TSize<T2>() const
	{
		return TSize<T2>(width, height);
	};

	//Comparisons
	/**
	 * Equals
	 */
	inline bool operator== (const TSize<T> &aSz) const
	{
		return ( (width == aSz.width) && (height == aSz.height) );
	};

	/**
	 * Not equals
	 */
	inline bool operator!= (const TSize<T> &aSz) const
	{
		return ( (width != aSz.width) || (height != aSz.height) );
	};

	/**
	 * Larger than with regards to size
	 * @param aSz size to compare with
	 * @return true if this size has larger area than aSz
	 */
	inline bool operator > (const TSize<T> &aSz) const
	{
		return (area() > aSz.area());
	};

	/**
	 * Larger than or equal with regards to size
	 * @param aSz size to compare with
	 * @return true if this size has larger or equal area than aSz
	 */
	inline bool operator >= (const TSize<T> &aSz) const
	{
		return (area() >= aSz.area());
	};

	/**
	 * Smaller than with regards to size
	 * @param aSz size to compare with
	 * @return true if this size has smaller area than aSz
	 */
	inline bool operator < (const TSize<T> &aSz) const
	{
		return (area() < aSz.area());
	};

	/**
	 * Smaller than or equals with regards to size
	 * @param aSz size to compare with
	 * @return true if this size has smaller or equal area than aSz
	 */
	inline bool operator <= (const TSize<T> &aSz) const
	{
		return (area() <= aSz.area());
	};

	//Other operations
	/**
	 * Area
	 * @return area of this size
	 */
	inline T area() const
	{
		return width * height;
	};

	/**
	 * Circumference
	 * @return circumference of this size
	 */
	inline T circumference() const
	{
		return 2 * (width + height);
	};
};

/// Two-dimensional rectangle template
template <typename T>
struct TRect
{
	/** abscissa of the top-left corner of the rectangle */
	T x;
	/** ordinate of the top-left corner of the rectangle */
	T y;
	/** width of the rectangle */
	T width;
	/** height of the rectangle */
	T height;

	/**
	 * Constructor
	 * @param x abscissa of top-left point
	 * @param y ordinate of top-left point
	 * @param w width
	 * @param h height
	 */
	template <typename T2, typename T3, typename T4, typename T5>
	explicit TRect(T2 x_, T3 y_, T4 w, T5 h) :
		x(cast(x_)),
		y(cast(y_)),
		width(cast(w)),
		height(cast(h))
	{};

	/**
	 * Constructor
	 * @param x abscissa of top-left point
	 * @param y ordinate of top-left point
	 * @param w width
	 * @param h height
	 */
	template <typename T2, typename T3>
	explicit TRect(const TPoint2<T2> &aPt, const TSize<T3> &aSz) :
		x(cast(aPt.x)),
		y(cast(aPt.y)),
		width(cast(aSz.width)),
		height(cast(aSz.height))
	{};

	///Constructor
	explicit TRect() : x(0), y(0), width(0), height(0) {};

	//Arithmetic operators
	/**
	 * Addition to size - changes rectangle's size
	 */
	template <typename T2>
	TRect<typename promote<T,T2>::result> operator+ (const TSize<T2> &aSz) const
	{
		return TRect<typename promote<T,T2>::result>(x, y, width + aSz.width, height + aSz.height);
	};

	/**
	 * Addition & assignment of size
	 */
	template <typename T2>
	TRect<T>& operator+= (const TSize<T2> &aSz)
	{
		width += aSz.width;
		height+= aSz.height;
		return *this;
	};

	/**
	 * Addition to point - shifts rectangle
	 */
	template <typename T2>
	TRect<typename promote<T,T2>::result> operator+ (const TPoint2<T2> &aPt) const
	{
		return TRect<typename promote<T,T2>::result>(x+aPt.x, y+aPt.y, width, height);
	};

	/**
	 * Addition & assignment of point - shifts rectangle
	 */
	template <typename T2>
	TRect<T>& operator+= (const TPoint2<T2> &aPt)
	{
		x += aPt.x;
		y += aPt.y;
		return *this;
	};

	//Typecast
	/**
	 * Typecasts with rounding
	 */
	template <typename T2>
	operator TRect<T2>() const
	{
		return TRect<T2>(x, y, width, height);
	};

	//Comparisons
	/**
	 * Equals
	 */
	bool operator== (const TSize<T> &aSz) const {return ( (width == aSz.width) && (height == aSz.height) ); };

	/**
	 * Not equals
	 */
	bool operator!= (const TSize<T> &aSz) const {return ( (width != aSz.width) || (height != aSz.height) ); };

	//Other operations
	/**
	 * Top left corner of rectangle
	 * @return top left corner of rectangle
	 */
	inline TPoint2<T> tl() const {return TPoint2<T>(x, y); };

	/**
	 * Bottom right corner of rectangle (non-inclusive)
	 * @return bottom right corner of rectangle. This point is considered not included in the rectangle
	 */
	inline TPoint2<T> br() const {return TPoint2<T>(x + width, y + height); };

	/**
	 * Get size of rectangle
	 * @return size of the rectangle
	 */
	inline TSize<T> size() const {return TSize<T>(width, height); };

	/**
	 * Area
	 * @return area of this size
	 */
	T area() const {return width * height; };

	/**
	 * Circumference
	 * @return circumference of this size
	 */
	T circumference() const {return 2 * (width + height); };

	template<typename T2>
	inline bool contains(const TPoint2<T2> &aPt) const
	{
		return ( (x <= aPt.x) && (x + width > aPt.x) && (y <= aPt.y) && (y + height > aPt.y) );
	}
};

/**
 * Class template that actually holds the matrix data.
 * This is used indirectly by the TMatrix class.
 */
template <typename T>
struct TMatrixData
{
	/// number of rows
	TUInt rows;
	/// number of columns
	TUInt cols;
	/// data
	std::vector<T> data;
	/**
	 * Constructor
	 * @param M #rows
	 * @param N #columns
	 */
	explicit TMatrixData(TUInt M, TUInt N) :
		rows(M),
		cols(N),
		data(M*N)
	{
	};

	/**
	 * Constructor
	 * @param M #rows
	 * @param N #columns
	 * @param val initialization value
	 */
	explicit TMatrixData(TUInt M, TUInt N, const T &val) :
		rows(M),
		cols(N),
		data(M*N, val)
	{
	};

	/**
	 * Constructor with initialization
	 * @param M #rows
	 * @param N #columns
	 * @param vals a vector containing initial values for the matrix
	 */
	explicit TMatrixData(TUInt M, TUInt N, const T* vals):
		rows(M),
		cols(N),
		data(vals, vals + M*N)
	{
	}

	/// row access operators
	T* row(TUInt y) {return &(data[y * cols]); };
	const T* row(TUInt y) const {return &(data[y * cols]); };
	/// element access operators
	T& operator ()(TUInt row, TUInt col) {return data[row * cols + col]; };
	const T& operator ()(TUInt row, TUInt col) const {return data[row * cols + col]; };
	T& at(TUInt row, TUInt col) {return data[row * cols + col]; };
	const T& at(TUInt row, TUInt col) const {return data[row * cols + col]; };
};

template <>
struct TMatrixData<bool>
{
	/// number of rows
	TUInt rows;
	/// number of columns
	TUInt cols;
	/// data
	std::vector<TUInt8> data;
	/**
	 * Constructor
	 * @param M #rows
	 * @param N #columns
	 */
	explicit TMatrixData(TUInt M, TUInt N) :
		rows(M),
		cols(N),
		data(M*N)
	{
	};

	/**
	 * Constructor
	 * @param M #rows
	 * @param N #columns
	 * @param val initialization value
	 */
	explicit TMatrixData(TUInt M, TUInt N, bool val) :
		rows(M),
		cols(N),
		data(M*N, (val ? 1 : 0))
	{
	};

	/**
	 * Constructor with initialization
	 * @param M #rows
	 * @param N #columns
	 * @param vals a vector containing initial values for the matrix
	 */
	explicit TMatrixData(TUInt M, TUInt N, bool *vals):
		rows(M),
		cols(N),
		data(M*N)
	{
		for (TUInt i = 0; i < M*N; i++)
			data[i] = (vals[i] ? 1 : 0);
	}

	/// row access operators
	TUInt8* row(TUInt y) {return &(data[y * cols]); };
	const TUInt8* row(TUInt y) const {return &(data[y * cols]); };
	/// element access operators
	TUInt8& operator ()(TUInt row, TUInt col) {return data[row * cols + col]; };
	bool operator ()(TUInt row, TUInt col) const {return data[row * cols + col] > 0; };
	TUInt8& at(TUInt row, TUInt col) {return data[row * cols + col]; };
	bool at(TUInt row, TUInt col) const {return data[row * cols + col] > 0; };
};

//Matrix type
//this is similar to the opencv Mat_ template, but much more basic
//For more complicated operations, use opencv types instead.
template <typename T>
class TMatrix
{
public:
	/// number of rows
	TUInt rows;
	/// number of columns
	TUInt cols;
protected:
	/// Actual matrix data
	std::shared_ptr<TMatrixData<T> > matrixData_;
	/// data offset when this matrix point to a piece of another
	TUInt dataOffset;
public:
	/// Pointer to first member of data
	TUInt8* data;
	/**
	 * Constructor
	 * @param M #rows
	 * @param N #columns
	 */
	explicit TMatrix(TUInt M, TUInt N) :
		rows(M),
		cols(N),
		matrixData_(new TMatrixData<T>(M,N)),
		dataOffset(0),
		data((TUInt8*) &(matrixData_->data.front()))
	{
	};

	/**
	 * Constructor
	 * @param M #rows
	 * @param N #columns
	 */
	explicit TMatrix(TUInt M, TUInt N, const T& val) :
		rows(M),
		cols(N),
		matrixData_(new TMatrixData<T>(M, N, val)),
		dataOffset(0),
		data((TUInt8*) &(matrixData_->data.front()))
	{};

	/**
	 * Constructor
	 * @param M #rows
	 * @param N #cols
	 */
	explicit TMatrix(TUInt M, TUInt N, const T* initVals):
		rows(M),
		cols(N),
		matrixData_(new TMatrixData<T>(M, N, initVals)),
		dataOffset(0),
		data((TUInt8*) &(matrixData_->data.front()))
	{};

	/**
	 * Constructor
	 * @param M #rows
	 * @param N #columns
	 */
	explicit TMatrix() :
		rows(0),
		cols(0),
		matrixData_(NULL),
		dataOffset(0),
		data(NULL)
	{
	};

	/**
	 * Constructor
	 * @param aSz size of matrix
	 */
	explicit TMatrix(const TSize<TUInt> &aSz) :
		rows(aSz.height),
		cols(aSz.width),
		matrixData_(new TMatrixData<T>(aSz.height, aSz.width)),
		dataOffset(0),
		data((TUInt8*) &(matrixData_->data.front()))
	{};

	/**
	 * Constructor
	 * @param aSz size of matrix
	 * @param val value to initialize matrix with
	 */
	explicit TMatrix(const TSize<TUInt> &aSz, const T& val) :
		rows(aSz.height),
		cols(aSz.width),
		matrixData_(new TMatrixData<T>(rows, cols, val)),
		dataOffset(0),
		data((TUInt8*) &(matrixData_->data.front()))
	{};

	/**
	 * Constructor
	 * @param aSz size of matrix
	 * @param initVals array of values to initialize matrix with
	 */
	explicit TMatrix(const TSize<TUInt> &aSz, const T* initVals):
		rows(aSz.height),
		cols(aSz.width),
		matrixData_(new TMatrixData<T>(rows, cols, initVals)),
		dataOffset(0),
		data((TUInt8*) &(matrixData_->data.front()))
	{};

	explicit TMatrix(const TMatrix<T> &aMat, const TRect<TUInt> &roi) :
		rows(roi.height),
		cols(roi.width),
		matrixData_(aMat.matrixData_),
		dataOffset(aMat.dataOffset + roi.y * matrixData_->cols + roi.x),
		data((TUInt8*) &(matrixData_->data[dataOffset]))
	{};

	/// row access operators
	/**
	 * pointer to data in a given y
	 * @param y row index
	 * @return pointed to the data in row y
	 */
	T* operator [](TUInt y)
	{
		assert (y < rows);
		return ((T*) data) + y * matrixData_->cols;
	};
	/**
	 * pointer to data in a given y
	 * @param y row index
	 * @return pointed to the data in row y
	 */
	const T* operator [](TUInt y) const
	{
		assert (y < rows);
		return ((T*) data) + y * matrixData_->cols;
	};
	/**
	 * pointer to data in a given row
	 * @param y row index
	 * @return pointed to the data in row y
	 */
	T* ptr(TUInt y)
	{
		assert (y < rows);
		return ((T*) data) + y * matrixData_->cols;
	};
	/**
	 * pointer to data in a given row
	 * @param y row index
	 * @return pointed to the data in row y
	 */
	const T* ptr(TUInt y) const
	{
		assert (y < rows);
		return ((T*) data) + y * matrixData_->cols;
	};

	/// element access operators
	/**
	 * retrieves data in a given row and column
	 * @param row row index
	 * @param col column index
	 * @return value of data in (row, column)
	 */
	T& operator ()(TUInt row, TUInt col)
	{
		assert ((col < cols) && (row < rows));
		return *((T*) data + row * matrixData_->cols + col);
	};
	/**
	 * retrieves data in a given row and column
	 * @param row row index
	 * @param col column index
	 * @return value of data in (row, column)
	 */
	const T& operator ()(TUInt row, TUInt col) const
	{
		assert ((col < cols) && (row < rows));
		return *((T*) data + row * matrixData_->cols + col);
	};
	/**
	 * retrieves data in a given row and column
	 * @param pt coordinates of data
	 * @return value of data in (pt.y, pt.x)
	 */
	T& operator ()(const TPoint2<TUInt> &pt)
	{
		assert ((pt.x < cols) && (pt.y < rows));
		return *((T*) data + pt.y * matrixData_->cols + pt.x);
	};
	/**
	 * retrieves data in a given row and column
	 * @param pt coordinates of data
	 * @return value of data in (pt.y, pt.x)
	 */
	const T& operator ()(const TPoint2<TUInt> &pt) const
	{
		assert ((pt.x < cols) && (pt.y < rows));
		return *((T*) data + pt.y * matrixData_->cols + pt.x);
	};
	/**
	 * retrieves data in a given row and column
	 * @param row row index
	 * @param col column index
	 * @return value of data in (row, column)
	 */
	T& at(TUInt row, TUInt col)
	{
		assert ((col < cols) && (row < rows));
		return *((T*) data + row * matrixData_->cols + col);
	};
	/**
	 * retrieves data in a given row and column
	 * @param row row index
	 * @param col column index
	 * @return value of data in (row, column)
	 */
	const T& at(TUInt row, TUInt col) const
	{
		assert ((col < cols) && (row < rows));
		return *((T*) data + row * matrixData_->cols + col);
	};
	/**
	 * retrieves data in a given row and column
	 * @param pt coordinates of data
	 * @return value of data in (pt.y, pt.x)
	 */
	T& at(const TPoint2<TUInt> &pt)
	{
		assert ((pt.x < cols) && (pt.y < rows));
		return *((T*) data + pt.y * matrixData_->cols + pt.x);
	};
	/**
	 * retrieves data in a given row and column
	 * @param pt coordinates of data
	 * @return value of data in (pt.y, pt.x)
	 */
	const T& at(const TPoint2<TUInt> &pt) const
	{
		assert ((pt.x < cols) && (pt.y < rows));
		return *((T*) data + pt.y * matrixData_->cols + pt.x);
	};

	/**
	 * True if data is reached in a continuous fashion, meaning there is no skipping at the end of a row
	 * @return true if data is continuous
	 */
	inline bool isContinuous() const {return (matrixData_->cols == cols); };

	//Misc. operations
	/**
	 * Clone - deep copies the data so that unlike the assignment operator, the clone does not share the same data.
	 * Furthermore, this allows a deep copy to be made at the same time as a cast.
	 * @return a deep copy of this matrix
	 */
	template <typename T2>
	TMatrix<T2> clone() const
	{
		TMatrix<T2> clonedMat(rows, cols);
		const T* inData = ((T*) data);
		for (TUInt i = 0; i < rows; i++)
		{
			for (TUInt j = 0; j < cols; j++)
				clonedMat(i,j) = cast<T,T2>(*(inData + j));
			inData += matrixData_->cols;
		}
		return clonedMat;
	};

	/**
	 * Clone - deep copies the data so that unlike the assignment operator, the clone does not share the same data.
	 * @return a deep copy of this matrix
	 */
	TMatrix<T> clone() const
	{
		if (isContinuous()) //fast copy
		{
			return TMatrix<T>(rows, cols, (T*) data);
		}
		else //slow copy
		{
			TMatrix<T> clonedMat(rows, cols);
			const T* inData = ((T*) data);
			for (TUInt i = 0; i < rows; i++)
			{
				for (TUInt j = 0; j < cols; j++)
					clonedMat(i,j) = *(inData + j);
				inData += matrixData_->cols;
			}
			return clonedMat;
		}
	};

	/**
	 * Extracts a region of interest
	 * @param roi a rectangle contained within this matrix
	 * @return a sub-matrix of this matrix determined by the roi rectangle
	 */
	TMatrix<T> operator()(const TRect<TUInt> &roi)
	{
		TMatrix<T> aMat(*this, roi);
		return aMat;
	}

	/**
	 * Extracts a region of interest
	 * @param roi a rectangle contained within this matrix
	 * @return a sub-matrix of this matrix determined by the roi rectangle
	 */
	const TMatrix<T> operator()(const TRect<TUInt> &roi) const
	{
		const TMatrix<T> aMat(*this, roi);
		return aMat;
	}

	/**
	 * True if the matrix is empty
	 * @return true if matrix is empty
	 */
	bool empty() const {return matrixData_->data.empty(); };

	/**
	 * Returns the size of this matrix
	 * @return size of this matrix
	 */
	TSize<TUInt> size() const {return TSize<TUInt>(cols, rows); };

	//TODO: matrix iterators
	//TODO: simple matrix operations

};

}; //end namespace ski

#endif // SKI_CV_TYPES_H_
