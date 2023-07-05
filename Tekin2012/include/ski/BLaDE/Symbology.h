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
 * @file Symbology.h
 * @author Ender Tekin
 *
 */

#ifndef SYMBOLOGY_H_
#define SYMBOLOGY_H_
#include <string>
#include <vector>
#include <list>
#include "ski/cv.hpp"

/** Matrix to store digit energies, energies_(digit, symbol)*/
typedef double TEnergy;
typedef TMatrixDouble TMatEnergy;

using namespace std;

/**
 * Class for constant-width barcode structures
 */
class BarcodeSymbology
{
public:
	/**
	 * Struct for expected barcode structure edges
	 */
	struct Edge
	{
		/**
		 * Constructor
		 * @param[in] i index of this edge
		 * @param[in] loc location of this edge
		 */
		Edge(int i, int loc) : index(i), location(loc) {};

		/** Index of the edge from the beginning */
		int index;

		/** Location of this edge */
		int location;

		/**
		 * Whether this is a fixed edge or not
		 * @return true if this edge's location is known (this is a fixed edge).
		 */
		inline bool isFixed() const {return (location != -1); };

		/** Polarity of this edge, must be one of -1 for even edges, 1 for odd edges */
		inline int polarity() const {return (index % 2 ? 1 : -1); };

		/** Number of positive edges before this edge */
		inline int nPreviousPositiveEdges() const {return index >> 1; };

		/** Number of negative edges before this edge */
		inline int nPreviousNegativeEdges() const {return (index % 2 ? (index >> 1) + 1 : index >> 1); };
	};

	struct Bar
	{
		/**
		 * Constructor
		 * @param[in] left pointer to left edge of this bar.
		 * @param[in] right pointer to left edge of this bar.
		 */
		inline Bar(Edge *left, Edge *right) : leftEdge(left), rightEdge(right), index(left->index) {};

		/** Polarity of the stripe, 1 for light stripes, -1 for dark stripes */
		inline bool isDark() const {return (leftEdge->polarity() == -1); };

		/** Left edge*/
		Edge *leftEdge;

		/** Right edge location */
		Edge *rightEdge;

		/** Index of the bar starting from the beginning */
		int index;

		/** Width of this stripe in terms of the fundamental width. -1 means unknown */
		inline int width() const { return ( leftEdge->isFixed() && rightEdge->isFixed() ? rightEdge->location - leftEdge->location : -1); };

	};

	/**
	 * Class for constant width barcode symbols
	 */
	struct Symbol
	{
		/**
		 * Constructor for a symbol with known pattern, such as guardbands
		 * @param[in] aWidth width of the symbol in multiples of the fundamental width
		 * @param[in] dataSymbolIndex index of data symbol. -1 if this is not a data symbol.
		 * @param[in] aBars vector representing the symbol bars.
		 */
		Symbol(TUInt aWidth, int dataSymbolIndex, vector<Bar*> aBars);

		/** Width of symbol. */
		TUInt width;

		/** index of this data symbol. -1 if this is a special symbol */
		int index;

		/** Pattern */
		vector<Bar*> bars;

		/**
		 * Left edge of this symbol
		 * @return pointer to left edge of the first bar of this symbol
		 */
		inline const Edge* leftEdge() const {return bars.front()->leftEdge; };

		/**
		 * Right edge of this symbol
		 * @return pointer to right edge of the last bar of this symbol
		 */
		inline const Edge* rightEdge() const {return bars.back()->rightEdge; };

		/** Whether this is a special symbol or a data symbol */
		inline bool isDataSymbol() const {return (index != -1); };
	};

	/**
	 * Constructor
	 * @param[in] name of symbology
	 * TODO: symbologies should only be held through handles -> move this to protected and add static create() method.
	 */
	BarcodeSymbology(const char name[]) : name_(name), nDataSymbols_(0), nFixedEdges_(0) {};

	/**
	 * Destructor
	 */
	virtual ~BarcodeSymbology() {};

	/**
	 * Name of symbology
	 * @return a string that is the symbology name
	 */
	inline const char* name() const {return name_; };

	/**
	 * Returns a reference to the requested fixed edge
	 * @param[in] i index of the fixed edge
	 * @return pointer to the requested fixed edge
	 */
	const Edge* getFixedEdge(TUInt i) const;

	/**
	 * Returns a reference to the requested data symbol
	 * @param[in] i index of the data symbol
	 * @return pointer to the requested symbol.
	 */
	const Symbol* getDataSymbol(TUInt i) const;

	/**
	 * Number of data symbols in the symbology
	 */
	inline TUInt nDataSymbols() const { return dataSymbols_.size(); };

	/**
	 * Number of fixed edges in the symbology
	 */
	inline TUInt nFixedEdges() const {return fixedEdges_.size(); };

	/**
	 * Number of total edges in the symbology
	 */
	inline TUInt nTotalEdges() const {return edges_.size(); };

	/**
	 * Width of the symbology in terms of the fundamental width
	 */
	inline TUInt width() const {return edges_.back().location; };

	/**
	 * Will return a convolution pattern for the particular symbology. Must be overwritten by the derived symbology
	 */
	virtual void getConvolutionPattern(TUInt digit, double x, bool isFlipped, vector<TUInt> &pattern) const;

	/**
	 * Final joint decoding of the barcode from digit energies.
	 * This method must be overwritten by the specific symbologies.
	 * @param[in] energy matrix, where energies[s,d] corresponds to the energy of the s'th symbol being digit d
	 * @return  a string that is the barcode estimate, empty string if no verified estimate is made
	 */
	virtual string estimate(const TMatEnergy &energies) const = 0;

	/**
	 * Convert estimated symbols to a string
	 * @param[in] aEstimate estimate of most likely symbols from given symbol energies,
	 * corresponding to the row indices in the energy matrix.
	 * @return a string that represents the vector of symbols in textual form
	 */
	virtual string convertEstimateToString(const vector<TUInt> &aEstimate) const;

protected:
	//Symbology name
	const char *name_;

	//Barcode representation:
	/** Representation of the barcode as a vector of edges.*/
	list<Edge> edges_;

	/** Representation of the barcode as a vector of bars.*/
	list<Bar> bars_;

	/** Representation of the barcode as a vector of symbols.*/
	list<Symbol> symbols_;

	/** Pointers to the fixed edges of the barcode */
	vector<Edge*> fixedEdges_;

	/** Pointers to the data symbols of the barcode */
	vector<Symbol*> dataSymbols_;

	/** Number of data symbols in this symbology */
	TUInt nDataSymbols_;

	/** Number of fixed edges in this symbology */
	TUInt nFixedEdges_;

	/**
	 * Adds a symbol to the barcode symbol representation
	 * @param[in] width width of the symbol in multiples of the fundamental width
	 * @param[in] nBars Number of bars+gaps in the symbol
	 * @param[in] pattern pointer to array representing the symbol pattern. NULL if unknown.
	 */
	Symbol* addSymbol(TUInt width, TUInt nBars, const TUInt *pattern=NULL);

private:
	/**
	 * Adds a bar to the barcode bar representation
	 * @param[in] rightEdgeLocation location of the right edge if known, -1 if unknown.
	 */
	Bar* addBar(int rightEdgeLocation);

	/**
	 * Adds an edge to the barcode edge representation
	 * @param[in] loc location of the edge.
	 */
	Edge* addEdge(int loc);
};



#endif /* SYMBOLOGY_H_ */
