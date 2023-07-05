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
 * Defines a Viterbi class which can be used to set up and run a Viterbi algorithm.
 * @author Ender Tekin
 */

#ifndef VITERBI_H
#define VITERBI_H

#include <vector>
#undef max
#undef min
#include "ski/math.h"
#include "ski/types.h"
#include <stdexcept>
#include <algorithm>

/*WARNING:  If the type is a bounded type (such as int), overflows may give incorrect results.  
Thus, it is recommended to use a larger type than strictly necessary (e.g. int instead of int8 etc.);
*/
using namespace std;

//=================================
//Implements the Viterbi algorithm
//=================================
/** 
 * Class declaration for minimum Energy Viterbi Path detection
 */
template <typename T=double>
class Viterbi
{
public:
	/**
	 * Return type for Viterbi - holds the energy and path of a sequence.
	 */
	typedef vector<T> TArray_;
#if CAN_USE_TEMPLATE_ALIAS
	typedef ski::TMat<T> TMatrix_;
#elif defined USE_OPENCV //TODO: once template aliases are implemented (gcc4.7+), use template aliases in cv.hpp instead
	typedef cv::Mat_<T> TMatrix_;
#else
	typedef ski::TMatrix<T> TMatrix_;
#endif

	struct Solution
	{
		/** energy of solution */
		T energy;
		/** sequence of state indices for this solution */
		vector<int> sequence;
		/**
		 * Initialization
		 */
		Solution (): energy((T) 0) {};
		/**
		 * Initialization with the expected path length
		 * @param[in] n length of path.
		 */
		Solution (int n): energy((T) 0), sequence(n, 0) {};
	};

	struct SubState
	{
		/** energy of this state */
		T energy;
		/** index of this state */
		int index;
		/** Path of this state */
		int path;
		/** Previous state in the path to this state */
		const SubState *prevState;
		/** Constructor */
		SubState(int i, int j=0, T e=((T) 0), SubState *pS=NULL);
		/** Copy constructor */
		SubState(const SubState &aSubState);
		/** For sorting states */
		inline bool operator< (const SubState& anotherState) const {return energy < anotherState.energy; };        
	};

	struct State
	{
		/** index */
		int index;
		/** Number of paths in this state */
		int nPaths;
		/** substates of this state */
		vector<SubState> substates;
		/**
		 * Constructor
		 * @param[in] i index of the state
		 * @param[in] n number of contained substates
		 */
		State(int i, int n=1);
		/** Destructor */
		~State() {};
		/**
		 * Calculates and assigns energies from priors only
		 * @param[in] prior prior for this time and state
		 */
		void calculate(T prior);
		/**
		 * Calculates and assigns energies using energies of previous states, prior and conditionals
		 * @param[in] prevStates energies of the states at the previous time
		 * @param[in] prior prior for this time
		 * @param[in] conditionals vector of conditional energies at this time given the states at the previous time
		 */
		void calculate(const vector<State> &prevStates, T prior, const TMatrix_ &conditionals);
	};

	struct Variable
	{
		/** States of this variable */
		vector<State> states;
		/**
		 * Constructor
		 * @param[in] t index of this variable
		 * @param[in] n number of states of this variable
		 */
		Variable(int t, int n);
		/** Destructor */
		~Variable() {};
		/** Index of this variable */
		int index;
		/** Number of paths to track for each state */
		int nPaths;
		/**
		 * Resizes the variable to have n states
		 * @param[in] n number of states the variable can have
		 */
		void resize(int n);
		/**
		 * Calculates the state energies from the prior
		 */
		void calculate(const vector<T> &prior);
		/**
		 * Calculates the state energies from the previous variable, prior and conditions
		 * @param[in] prevVar previous variable
		 * @param[in] prior prior energies
		 * @param[in] conditional conditional energies
		 */
		void calculate(const Variable &prevVar, const TArray_ prior, const TMatrix_ conditional);
		/**
		 * Finds the best nPaths substates and returns references to them.
		 * @param[out] beststates references to the best substates
		 */
		void bestStates(vector<SubState> &beststates);
	};

	/**
	 * Constructor
	 * @param[in] sE singleton energies.  sE[t](i) is the energy of state i at time t.
	 * @param[in] pE pairwise energies.  pE[t](i,j) is the energy to move from state i at time t to state j at time t+1.
	 * @param[in] nPaths number of paths to return.
	 *
	 */
    Viterbi(const vector<TArray_> &sE, const vector<TMatrix_> &pE, int nPaths=1);

	/**
	 * Destructor
	 */
    ~Viterbi();

	/**
	 * Solves the Viterbi algorithm.
	 * @param[in] finalState index of the final state to end. By default, just finds the best state.
	 * If given a valid index, backtracks from that specific state.
	 */
	void solve(int finalState=-1);

	/** Holds the solutions to the Viterbi, sorted in order of increasing energy*/
	vector<Solution> solutions;
private:
	/** 
	 * Runs the algorithm
	 */
	void run();

	/**
	 * backtracks from pseudostate finalState
	 * @param[in] finalState final state to backtrack from
	 */
	void backtrack(int finalState);

	/**
	 * Checks the consistency of the inputs and prints errors if there is a problem.
	 * Also initializes the states.
	 * @return true if the inputs are of consistent size.
	 */
	void initialize();

	/** Reference to the singleton energies.  [t](i) is the prior energy of state i at time t */
	const vector<TArray_> &priors_;
	/** Reference to the pairwise energies.  conditionals_[t](i,j) is the conditional energy from i to j at time t */
	const vector<TMatrix_> &conditionals_;
	/** sequence length */
    int time_;
	/** number of paths to track */
	int nPaths_;

	/**States*/
	vector<Variable> vars_;
};

//==================
// Substates
//===================

template <typename T>
Viterbi<T>::SubState::SubState(int i, int j/*=0*/, T e/*=((T) 0)*/, SubState *pS/*=NULL*/):
	energy(e),
	index(i),
	path(j),
	prevState(pS)
{
}

template <typename T>
Viterbi<T>::SubState::SubState(const SubState &aSubState):
	energy(aSubState.energy),
	index(aSubState.index),
	path(aSubState.path),
	prevState(aSubState.prevState)
{
}

//==================
// States
//===================
template <typename T>
Viterbi<T>::State::State(int i, int n/*=1*/):
	index(i),
	nPaths(n),
	substates(n, SubState(i))
{
}

//==================
// Variables
//===================
template <typename T>
Viterbi<T>::Variable::Variable(int t, int p):
	index(t),
	nPaths(p)
{
}

template <typename T>
void Viterbi<T>::Variable::resize(int nStates)
{
	//Add states if necessary
	states.reserve(nStates);
	for (int n = states.size(); n < nStates; n++)
		states.push_back(State(n, nPaths));
	//trim states if too many
	states.erase(states.begin() + nStates, states.end());
}

template <typename T>
void Viterbi<T>::Variable::calculate(const TArray_ &prior)
{
	resize(prior.size());
	typename vector<T>::const_iterator p = prior.begin();
	for (typename vector<State>::iterator s = states.begin(); s != states.end(); s++, p++)
	{
		int path = 0;
		for (typename vector<SubState>::iterator ss = s->substates.begin(); ss!= s->substates.end(); ss++)
		{
			ss->energy = (ss == s->substates.begin() ? *p : ski::MaxValue<T>() / 2);//*p;
			ss->path = path++;
		}
	}
}

template <typename T>
void Viterbi<T>::Variable::calculate(const Variable &prevVar, const TArray_ prior, const TMatrix_ conditional)
{
	if (prior.size() != states.size())
		resize(prior.size());
	if ( (conditional.rows != (int) prevVar.states.size()) || (conditional.cols != (int) states.size()) )
		throw logic_error("Viterbi::Variable: Size mismatch.");
	//For each state, calculate the min energy paths
	static vector<SubState> allPaths;
	int nAllPaths = prevVar.states.size() * prevVar.nPaths;
	allPaths.resize(nAllPaths, SubState(0));
	int n = 0;
	for (typename vector<State>::iterator s = states.begin(); s != states.end(); s++, n++)
	{
		T statePriorEnergy = prior[n];
		typename vector<SubState>::iterator p = allPaths.begin();
		int pN = 0;	//index of previous state
		for (typename vector<State>::const_iterator pS = prevVar.states.begin(); pS != prevVar.states.end(); pS++, pN++)
		{
			T stateTotalEnergy = statePriorEnergy + conditional(pN, n);
			for (typename vector<SubState>::const_iterator pSS = pS->substates.begin(); pSS != pS->substates.end(); pSS++, p++)
			{
				p->prevState = &(*pSS);
				p->energy = stateTotalEnergy + pSS->energy;
			}
		}
		partial_sort_copy(allPaths.begin(), allPaths.end(), s->substates.begin(), s->substates.end());
		//Fix path numbers
		int i = 0;
		for (typename vector<SubState>::iterator ss = s->substates.begin(); ss != s->substates.end(); ss++)
		{
			ss->index = n;
			ss->path = i++;
		}
	}
}

template <typename T>
void Viterbi<T>::Variable::bestStates(vector<SubState> &beststates)
{
	//For each state, calculate the min energy paths
	int nAllStates= states.size() * nPaths;
	beststates.clear();
	beststates.reserve(nAllStates);
	for (typename vector<State>::iterator s = states.begin(); s != states.end(); s++)
	{
		for (typename vector<SubState>::iterator ss = s->substates.begin(); ss != s->substates.end(); ss++)
			beststates.push_back(*ss);
	}
	partial_sort(beststates.begin(), beststates.begin() + nPaths, beststates.end());
	beststates.resize(nPaths, SubState(-1));
}

//==================
//VITERBI
//===================

template <typename T>
Viterbi<T>::Viterbi(const vector<TArray_ > &priorMat, const vector<TMatrix_> &condMat, int nPaths/*=1*/):
	solutions(nPaths),
	priors_(priorMat),
	conditionals_(condMat),
	nPaths_(nPaths)
{
}

template <typename T>
Viterbi<T>::~Viterbi()
{
}

template <typename T>
void Viterbi<T>::initialize()
{
	/*Checks that the sizes of the priors and conditionals are consistent with each other.
	Also assigns the time_ and nStates_ variables;
	Priors_[t] gives the size of nStates_[t],
	Conditionals_[t-1] nStates_[t-1] x nStates_[t]
	*/
	time_ = priors_.size();
	//Check lengths:
	if ((int) conditionals_.size() != time_-1)
		throw logic_error("Viterbi: Time inconsistency!");
	//Check that the matrices are compatible
	for (int t = 1; t < time_; t++)
	{
		if ( (conditionals_[t-1].rows != priors_[t-1].size())
				|| (conditionals_[t-1].cols != priors_[t].size()) )
			throw logic_error("Viterbi: Sizes of the provided matrices are not consistent!");
	}
	//Add states if need be
	vars_.reserve(time_);
	for (int t = vars_.size(); t < time_; t++)
		vars_.push_back(Variable(t, nPaths_));
	vars_.erase(vars_.begin() + time_, vars_.end());
	/** Initialize solutions */
	for (typename vector<Solution>::iterator s = solutions.begin(); s!= solutions.end(); s++)
	{
		s->energy = (T) 0;
		s->sequence.resize(time_);
	}
}

template <typename T>
void Viterbi<T>::solve(int finalState/*=-1*/)
{
    /* Returns nPaths best (min energy) Viterbi sequences ending at finalState
     * and their corresponding energies. */
	//Ensure finalState is valid:
	if ( (finalState >= (int) priors_.back().size()) || (finalState < -1) )
		throw invalid_argument("Viterbi: Final state not valid.");
	try
	{
		//Initialize states
		initialize();
		//Run the Viterbi
	    run();
	    //Backtrack
	    backtrack(finalState);
	}
	catch (exception &aErr)
	{
		throw aErr;
	}
}

template <typename T>
void Viterbi<T>::run()
{
	vars_[0].calculate(priors_[0]);
	for (int t = 1; t < time_; t++)
		vars_[t].calculate(vars_[t-1], priors_[t], conditionals_[t-1]);
}

template <typename T>
void Viterbi<T>::backtrack(int finalState)
{
	//Backtracks from the final state given in argument
	Variable &lastVar = vars_[time_-1];
	vector<SubState> finalStates;
	const SubState *aState;
	if (finalState == -1)
		lastVar.bestStates(finalStates);
	else
		finalStates.assign(lastVar.states[finalState].substates.begin(), lastVar.states[finalState].substates.end());
	for (int n = 0; n < nPaths_; n++)
	{
		aState = &finalStates[n];
		solutions[n].energy = aState->energy;
		for (int t = time_-1; t >= 0; t--)
		{
			solutions[n].sequence[t] = aState->index;
			aState = aState->prevState;
		}
	}
}

#ifdef _LIBTEST

void testViterbi()
{
	typedef int curType; //change curType to test different types
	int T = 4, nStates = 3;
	vector<vector<curType> > prior;
	vector<Mat_<curType> > cond;
	for (int t = 0; t < T; t++)
		prior.push_back(vector<curType>(nStates));
	for (int t = 0; t < T - 1; t++)
		cond.push_back(Mat_<curType>(nStates, nStates));

	cond[0](0,0) = 1;
	cond[0](0,1) = 0;
	cond[0](0,2) = 1;
	cond[0](1,0) = 0;
	cond[0](1,1) = 1;
	cond[0](1,2) = 2;
	cond[0](2,0) = 1;
	cond[0](2,1) = 2;
	cond[0](2,2) = 1;
	cond[1](0,0) = 0;
	cond[1](0,1) = 1;
	cond[1](0,2) = 1;
	cond[1](1,0) = 1;
	cond[1](1,1) = 2;
	cond[1](1,2) = 1;
	cond[1](2,0) = 3;
	cond[1](2,1) = 0;
	cond[1](2,2) = 1;
	cond[2](0,0) = 2;
	cond[2](0,1) = 2;
	cond[2](0,2) = 1;
	cond[2](1,0) = 3;
	cond[2](1,1) = 1;
	cond[2](1,2) = 0;
	cond[2](2,0) = 1;
	cond[2](2,1) = 0;
	cond[2](2,2) = 2;
	prior[0][0] = 1;
	prior[0][1] = 1;
	prior[0][2] = 2;
	prior[1][0] = 2;
	prior[1][1] = 1;
	prior[1][2] = 0;
	prior[2][0] = 1;
	prior[2][1] = 1;
	prior[2][2] = 2;
	prior[3][0] = 0;
	prior[3][1] = 1;
	prior[3][2] = 1;
	int nPaths = 4;
	Viterbi<curType> V(prior, cond, nPaths);
	V.solve();
	for (int p = 0; p < nPaths; p++){
		printf("%d'th best path: ",p);
		for (int t=0; t<4; t++)
			printf("%d,", V.solutions[p].sequence[t]);
		printf(" with energy %d.\n", V.solutions[p].energy); //change %d to %f if curType==float etc.
	}

	int finalState = 1;
	V.solve(finalState);
	for (int p = 0; p < nPaths; p++){
		printf("%d'th best path ending at %d: ", p, finalState);
		for (int t=0; t<4; t++)
			printf("%d,", V.solutions[p].sequence[t]);
		printf(" with energy %d.\n", V.solutions[p].energy); //change %d to %f if curType==float etc.
	}
	return;
	//Correct results for this test:
	//V.minEnergy(4);
	//[4 5 5 5]
	//V.bestSequence(4);
	//[0 2 1 2; 0 2 1 1; 1 2 1 2; 2 2 1 2] (sequences are rows)
	//V.minEnergy(4,1);
	//[5 6 6 6]
	//V.bestSequence(4,1);
	//P=[0 2 1 1; 0 1 2 1; 1 2 1 1; 0 2 2 1] (sequences are rows)
}
#endif //LIBTEST

#endif // VITERBI_H
