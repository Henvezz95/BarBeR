/*
 * ProductSearch.cpp
 *
 *  Created on: Sep 18, 2012
 *      Author: kamyon
 */

#include "ProductSearch.h"
#include <stdexcept>
#include <cstring>
#include "ski/log.h"

const char* ProductSearch::USER_AGENT = "Mozilla/4.0 (compatible; MSIE 5.01; Windows NT 5.0)";


ProductSearch::ProductSearch()
{
	//Initialize curl
	curl_global_init(CURL_GLOBAL_ALL);
}

ProductSearch::~ProductSearch()
{
	/* Cleanup curl*/
	curl_global_cleanup();
}

ProductSearch* ProductSearch::create(Method method)
{
	ProductSearch* searchMethod(NULL);
	switch (method)
	{
	case Google_Product_Search:
		searchMethod = new GoogleSearch();
		break;
	case Directions_For_Me:
		//searchMethod.set(new D4MSearch());
		//searchMethod = new D4MSearch();
		LOGE("Requested method removed");
		break;
	default:
		LOGE("Requested method not yet implemented\n");
		break;
	}
	return searchMethod;
}

ProductSearch::ProductList ProductSearch::identify(const std::string& bc)
{
	std::string page;
	//Initialize cURL request
	CURL* curl = initializeRequest();
	//prepare the request
	prepareRequest(curl, bc);
	//submit cURL request
	bool isSuccessfullySubmitted = submitRequest(curl, page);
	//Clean up after cURL
	cleanupAfterRequest(curl);
	//Now parse the page
	return (isSuccessfullySubmitted ? parseProductPage(page) : ProductList());
}

CURL* ProductSearch::initializeRequest()
{
	CURL* curl = curl_easy_init();
	if (!curl)
	{
		LOGE("Could not initialize cURL.");
		throw std::runtime_error("Cannot lookup product information");
	}

	//Set curl easy options
	curl_easy_setopt(curl, CURLOPT_USERAGENT, USER_AGENT);
	curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, true);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, getReturnedResult);
	curl_easy_setopt(curl, CURLOPT_WRITEHEADER, NULL);
#ifdef DEBUG_
		curl_easy_setopt(curl, CURLOPT_VERBOSE, true);
#endif
	return curl;
}

bool ProductSearch::submitRequest(CURL* curl, std::string &page)
{
	HttpResult htmlData;
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &htmlData);
	CURLcode res = curl_easy_perform(curl);
	// getReturnedResult() will be called when the result page arrives, and populate htmlData before returning
	if (res != CURLE_OK)
	{
		LOGD("cURL ERROR: %d\n", res);
		switch (res)
		{
		case CURLE_URL_MALFORMAT:
			LOGE("Search url is not correctly formatted.\n");
			break;
		case CURLE_COULDNT_RESOLVE_PROXY:
			LOGE("Could not resolve proxy server.\n");
			break;
		case CURLE_COULDNT_RESOLVE_HOST:
			LOGE("Could not resolve host.\n");
			break;
		case CURLE_COULDNT_CONNECT:
			LOGE("Could not connect to network.\n");
			break;
		case CURLE_REMOTE_ACCESS_DENIED:
			LOGE("Remote access denied - check credentials.\n");
			break;
		default:
			LOGE("Network error!\n");
			break;
		}
		LOGE("Product information cannot be determined\n");
		return false;
	}

	// Save the website to string
	page.assign((char*) htmlData.data, htmlData.size);
	LOGD("Product data retrieved\n");
	return true;
}

void ProductSearch::cleanupAfterRequest(CURL* curl)
{
	curl_easy_cleanup(curl);
}

size_t ProductSearch::getReturnedResult(void *buffer, size_t size, size_t nmemb, void *userp)
{
	//TODO: see if we can do this with just strings...
	HttpResult *htmlData = (HttpResult*) userp;
	size_t addedSize = size * nmemb;
	TUInt8 *addedData = htmlData->data + htmlData->size;	//beginning of new data
	if (htmlData->size + addedSize > HttpResult::BUFSIZE)
		addedSize = HttpResult::BUFSIZE - htmlData->size;
	std::memcpy(addedData, buffer, addedSize);
	htmlData->size += addedSize;
	if (htmlData->size == HttpResult::BUFSIZE)
		LOGW("Html buffer reached\n");
	return addedSize;
}
