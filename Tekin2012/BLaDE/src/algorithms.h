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
 * @file
 * Some Algorithms
 * @author Ender Tekin
 */

#ifndef EALGORITHMS_H_
#define EALGORITHMS_H_

#include "ski/math.h"
#include "ski/type_traits.h"
#include "ski/cv.hpp"
#include <vector>
using namespace std;

namespace ski
{


/** @class Vote template for voting algorithms */
template<class T1, class T2>
struct Vote_
{
	/** location of the vote */
	T1 loc;
	/** weight of the vote */
	T2 weight;
	/**
	 * Constructor
	 * @param[in] x location of the vote
	 * @param[in] w weight of the vote
	 */
	Vote_(T1 x=T1(), T2 w=T2()): loc(x), weight(w) {};
};


/** Some commonly used vote types */
typedef Vote_<double, double> Vote;
typedef Vote_<TPointInt, double> VoteP;

/** @class Kernel template class */
template<class T>
class Kernel
{
public:
	/** Returns the value of the kernel at a given location */
	virtual double value(T x) const = 0;
	/** Destructor */
	virtual ~Kernel() {};
};

/**
 * Evaluates the value of a point on a distribution based on weighted kde that wraps around.
 * @param[in] p vector of distribution weights, a kde is fitted to each point
 * @param[in] x the place to evaluate kde at.
 * @param[in] kernel kernel to use in the kde
 * @return the evaluated value
 */
template<class T1, class T2>
T2 kde(const vector<Vote_<T1, T2> > &p, const T1 x, const Kernel<T1> &kernel)
{
	typedef typename vector<struct Vote_<T1, T2> >::const_iterator VoteIterator;
	T2 w = T2();
	for (VoteIterator v = p.begin(); v != p.end(); v++)
		w += v->weight * kernel.value(v->loc - x);
	return w;
};

/**
 * Performs mean shift to find mode of distribution p assuming the distribution wraps around
 * @param[in] pIn vector of distribution weights and locations, a kde is fitted to each point
 * @param[in, out] pOut vector of modes and their weights. Contains the initial locations to perform mean shift on.
 * @param[in] kernel kernel to use in the kde
 */
template<class T1, class T2>
void meanShift(const vector<class Vote_<T1, T2> > &pIn, vector<class Vote_<T1, T2> > &pOut, const Kernel<T1> &kernel)
{
	TUInt n = pIn.size();
	pOut = pIn;
	if (n < 2)
		return;
	//generate weighted vector
	vector<Vote_<T1, T1> > pWeighted;
	pWeighted.reserve(n);
	typedef typename vector<struct Vote_<T1, T2> >::const_iterator VoteConstIterator;
	typedef typename vector<struct Vote_<T1, T2> >::iterator VoteIterator;
	for (VoteConstIterator x = pIn.begin(); x != pIn.end(); x++)
		pWeighted.push_back( Vote_<T1, T1>(x->loc, x->loc * x->weight) );
	static const TUInt maxIter = 100;
	static const double maxDist = 0.01;
	for (TUInt i = 0; i < maxIter; i++)
	{
		double totDistanceMoved = 0;
		for (VoteIterator x = pOut.begin(); x != pOut.end(); x++)
		{
			x->weight = kde(pIn, x->loc, kernel);
			T1 newLoc = kde(pWeighted, x->loc, kernel) * (1 / x->weight);
			totDistanceMoved += distance(x->loc, newLoc);
			x->loc = newLoc;
		}
		//check for convergence
		if (totDistanceMoved < maxDist)
			break;
	}
};

/**
 * Finds the centers of data that are a certain distance from each other
 */
template <class T1>
void findClusterCenters(const vector<class Vote_<T1, double> > &data, vector<class Vote_<T1, double> >  &centers, double radius)
{
	//ensure output is empty
	centers.clear();
	bool isMatchFound=false;
	typedef typename vector<class Vote_<T1, double> >::const_iterator VoteConstIterator;
	typedef typename vector<class Vote_<T1, double> >::iterator VoteIterator;
	VoteConstIterator v;
	VoteIterator v2;
	for (v = data.begin(); v!= data.end(); v++)
	{
		for (v2 = centers.begin(); v2 != centers.end(); v2++)
		{
			if ( ( isMatchFound = ( distance(v->loc, v2->loc) < radius ) ) ) //if matches an existing center
				break;
		}
		if (!isMatchFound)
			centers.push_back(*v);	//no match found, create new cluster center
		else
		{
			//Recalculate center and weight of the matching cluster
			double totWeight = v->weight + v2->weight;
			v2->loc = v2->loc * (v2->weight / totWeight) + v->loc * (v->weight / totWeight);
			v2->weight = totWeight;
		}
	}
};


/*
 * Specializations
 */

/** Gaussian kernel for reals */
class GaussianKernelD: public Kernel<double>
{
public:
	/**
	 * Constructor
	 * @param[in] var variance of the kernel
	 */
	GaussianKernelD(double var): z(1 / sqrt(2 * ski::PI * var)), c(-.5 / var) {};
	/**
	 * Returns the value of the kernel
	 * @param[in] d value to evaluate the kernel at
	 * @return value of the kernel at d.
	 */
	inline virtual double value(double d) const {return exp(c * d * d) / z; };
private:
	/** Constants for this kernel */
	const double z, c;
};

/**
 * @class Kernel used in meanshift in 2D space
  */
class GaussianKernelPt: public Kernel<TPointInt>
{
public:
	/**
	 * Constructor
	 * @param[in] var variance of the kernel
	 */
	GaussianKernelPt(double var): z(1 / sqrt(2 * PI * var)), c(-.5 / var) {};
	/**
	 * Returns the value of the kernel
	 * @param[in] d value to evaluate the kernel at
	 * @return value of the kernel at d.
	 */
	inline virtual double value(TPointInt d) const { return exp(c * norm(d)) / z; };
private:
	/** Constants for this kernel */
	const double z, c;
};

/**
 * @class Kernel used in kde that takes into account wraparound in 1D space
 */
class GaussianKernelRot: public Kernel<double>
{
public:
	/**
	 * Constructor
	 * @param[in] var variance of the kernel
	 * @param[in] maxVal maximum value of the kernel argument
	 */
	GaussianKernelRot(double var, double maxVal): z(1 / sqrt(2 * PI * var)), c(-.5 / var), lim(maxVal) {};
	/**
	 * Returns the value of the kernel
	 * @param[in] d value to evaluate the kernel at
	 * @return value of the kernel at d.
	 */
	inline virtual double value(double d) const
	{
		d = abs(d);
		if (d > lim)
			d = 2*lim-d;
		return exp(c * d * d) / z;
	};
private:
	/** Constants for this kernel */
	double z, c, lim;
};

} //end namespace E

#endif // EALGORITHMS_H_
