/*
 * ProductSearch_Google.cpp
 *
 *  Created on: Sep 20, 2012
 *      Author: kamyon
 */
#include "ProductSearch.h"
#include "ski/log.h"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>

using boost::property_tree::ptree;

GoogleSearch::GoogleSearch()
{
	LOGD("Initializing Google Product Search\n");
}

GoogleSearch::~GoogleSearch()
{
}

void GoogleSearch::prepareRequest(CURL* curl, const std::string &upc)
{
	// Form product search page
	std::string url = SEARCH_URL;
	url.replace(url.find("[KEY]"), 5, API_KEY);
	url.replace(url.find("[BARCODE]"), 9, upc);
	/* URL that receives this POST */
	curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
}

ProductSearch::ProductList GoogleSearch::parseProductPage(const std::string &page)
{
	LOGD("Parsing product data\n");

	ptree pTree, productDetailTree;
	std::stringstream pageStream(page);
	ProductList products;
	static const ptree::path_type path = "items..product";
	try
	{
		read_json(pageStream, pTree);
		productDetailTree = pTree.get_child("items.");
		for (ptree::const_iterator pBranch = productDetailTree.begin(); pBranch != productDetailTree.end(); pBranch++)
		{
			ptree::value_type val = *pBranch;
			ptree productDetailTree = val.second.get_child("product");
			ProductInfo info;
			info.title = productDetailTree.get<std::string>("title");
			info.brand = productDetailTree.get<std::string>("brand");
			info.description = productDetailTree.get<std::string>("description");
			LOGD("Product found: %s\n", info.title.c_str());
			products.push_back(info);
		}
	}
	catch (boost::property_tree::ptree_error &aErr)
	{
		LOGD("could not find product name on returned webpage\n");
	}
	return products;
}
///Pleace your API key here
const std::string GoogleSearch::API_KEY = "";
/** Search address */
const std::string GoogleSearch::SEARCH_URL = "https://www.googleapis.com/shopping/search/v1/public/products?key=[KEY]&country=US&q=[BARCODE]&alt=json&fields=items/product(title,brand,description)";





