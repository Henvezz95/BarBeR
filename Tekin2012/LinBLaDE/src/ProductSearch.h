/*
 * ProductSearch.h
 *
 *  Created on: Sep 18, 2012
 *      Author: kamyon
 */

#ifndef PRODUCTSEARCH_H_
#define PRODUCTSEARCH_H_

#include <string>
#include "ski/types.h"
#include <curl/curl.h>
#include <list>

class ProductSearch
{
protected:
	/**
	 * Constructor
	 */
	ProductSearch();

public:

	/**
	 * Destructor
	 * TODO: make protected after implementing the factory method as returning std::unique_ptr
	 * TODO: pImpl this.
	 */
	virtual ~ProductSearch();

	/**
	 * Search method
	 */
	enum Method
	{
		Google_Product_Search,//!< Google Product Search
		Directions_For_Me     //!< Directions For Me
	};

	/**
	 * Product Information
	 */
	struct ProductInfo
	{
		/** Name */
		std::string title;
		/** Description */
		std::string description;
		/** Manufacturer */
		std::string brand;
		/** Combines the information as a single string */
		std::string asString() const
		{
			std::string info;
			if (!title.empty())
				info += "Title: " + title + "\n";
			if (!brand.empty())
				info += "Brand: " + brand + "\n";
			if (!description.empty())
				info += "Description:" + description + "\n";
			return info;
		};
	};

	typedef std::list<ProductInfo> ProductList;

	static ProductSearch* create(Method method);

	/**
	 * Takes a barcode with identified upc and fills the product information
	 * @param[in] bc barcode to identify product from
	 * @return product information
	 */
	ProductList identify(const std::string &bc);

protected:
	/**
	 * Structure to store intermediate results of http requests
	 */
	struct HttpResult
	{
		/** Pointer to actual data */
		TUInt8 *data;
		/** Size of data */
		size_t size;
		/** buffersize */
		static const size_t BUFSIZE = 100000; //100K buffer
		/** Constructor */
		HttpResult() : data(new TUInt8[BUFSIZE]), size(0) {};
		/** Destructor */
		~HttpResult() {if (data != NULL) delete [] data; };
	};

	/** User agent to mask as */
	static const char* USER_AGENT;

	/**
	 * Initializes a cURL easy session, and sets up some common parameters
	 * @return handle to the initialized easy cURL session.
	 */
	static CURL* initializeRequest();

	/**
	 * Prepares the cURL request for the specific lookup method
	 * @param[in] curl cURL session handle
	 * @param[in] upc upc code to look up
	 */
	virtual void prepareRequest(CURL* curl, const std::string &upc) = 0;

	/**
	 * Submits a prepared request
	 * @param[in] curl cURL session handle
	 * @param[out] page a string containing the returned page that needs to be parsed.
	 * @return true if request has been successfully submitted, false if an error occurred.
	 */
	static bool submitRequest(CURL* curl, std::string &page);

	/**
	 * Cleans up after a cURL easy session
	 * @param[in] cURL session handle to cleanup after
	 */
	virtual void cleanupAfterRequest(CURL* curl);

	/**
	 * Callback function that saves the returned results webpage
	 * @param[in] buffer buffer that contains returned result
	 * @param[in] size size of result in units of size nmemb
	 * @param[in] nmemb size of each unit returned
	 * @param[in] pointer to user supplied field to store returned results.
	 */
	static size_t getReturnedResult(void *buffer, size_t size, size_t nmemb, void *userp);

	/**
	 * Parses returned page to extract product information
	 * @param[in] page html page to parse
	 * @return information about the product
	 */
	virtual ProductList parseProductPage(const std::string &page) = 0;
};


//==========================
// Google Search
//==========================
class GoogleSearch: public ProductSearch
{
public:
	GoogleSearch();
protected:

	/** API Key*/
	static const std::string API_KEY;

	/** API Key*/
	static const std::string SEARCH_URL;

	virtual ~GoogleSearch();

	void prepareRequest(CURL* curl, const std::string &upc);

	ProductList parseProductPage(const std::string &page);
};

//==========================
// D4M Search
//==========================

class D4MSearch: public ProductSearch
{
public:
	/** Constructor */
	D4MSearch();

protected:

	/** Search address */
	static const std::string SEARCH_URL;

	/** Referrer */
	static const std::string REFERRER;

	/** Destructor */
	virtual ~D4MSearch();

	void prepareRequest(CURL* curl, const std::string &upc);

	void cleanupAfterRequest(CURL* curl);

	ProductList parseProductPage(const std::string &page);

	struct curl_httppost *formpost;
	struct curl_httppost *lastptr;

};


#endif /* PRODUCTSEARCH_H_ */
