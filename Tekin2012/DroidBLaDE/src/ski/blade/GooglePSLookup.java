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
package ski.blade;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.DefaultHttpClient;
import org.json.JSONArray;
import org.json.JSONObject;


import android.util.Log;

public class GooglePSLookup implements UPCLookup.WebLookup {
	private final String searchUrlBase = "https://www.googleapis.com/shopping/search/v1/public/products?key=[KEY]&country=US&q=[BARCODE]&alt=json&fields=items/product(title,brand,description)";
	private final String googleApiKey = "";	//TODO: should we read this from resources?
	private final String TAG = "BLaDE Google Lookup";
	private boolean isLookingUp_;
		
	public String performLookup(String upc) {
		Log.d(TAG, "Performing web request for UPC:" + upc);
		String searchUrl = searchUrlBase.replace("[KEY]", googleApiKey).replace("[BARCODE]", upc);
		String result = null;
		if (isLookingUp_)
			Log.e(TAG, "A lookup is already under way");
		else {
			isLookingUp_ = true;
			try {
				result = webRequest(searchUrl);
				if (result != null)
					result = parseResult(result);
			}
			catch (Exception e) {
				Log.e(TAG,"Error performing lookup");
			}
			isLookingUp_ = false;
		}
		return result;
	}
	
	public void cancel() {
		isLookingUp_ = false;
	}
	
	/**
	 * Performs an Http request for barcode info and returns the resulting page/
	 * @param[in] upc upc string to look up
	 * @return web page returned as a string
	 * @throws Exception if the http connection cannot be established
	 */
	private String webRequest(String searchUrl) throws Exception {
		//Modified from "ProAndroid 2" by Hashimi, Komatineni and MacLean.
		BufferedReader in = null;
		int retries = 3;
		String result = null;
		//Create Http connection
		HttpClient client = new DefaultHttpClient();
		HttpGet request = new HttpGet(searchUrl);
		while ( isLookingUp_ && (retries > 0) ){
			try {
				//Execute Post request and receive response
				HttpResponse response = client.execute(request);
				in = new BufferedReader( new InputStreamReader(response.getEntity().getContent()));
				StringBuffer buf = new StringBuffer("");
				String line = null;
				String NL = System.getProperty("line.separator");
				while ((line = in.readLine()) != null)
					buf.append(line + NL);
				//Convert response to string
				result = buf.toString();
				retries = 0;	//Successful response 
			}
			catch (Exception e) {
				//If connection problem, try again until set limit
				if (retries > 0) {
					Log.w(TAG, "Error establishing http connection:" + e.getMessage() + " Retries remaining:" + retries);
					retries--;
				}
				else {
					Log.e(TAG, "Could not establish http connection:" + e.getMessage());
					throw e;
				}
			}
			finally {
				if (in != null) {
					try {
						in.close();
					}
					catch (IOException e) {
						e.printStackTrace();
					}
				}
			}
		}
		return result;
	}

	/**
	 * Parses a web page result of the http request and extracts important information
	 * @param page webpage returned from webRequest()
	 * @return barcode information in string format.
	 * Based on example from http://blog.andrewpearson.org/2010/07/android-html-parsing.html
	 */
	private String parseResult(String page) {
		String result = null;
		try {
			JSONObject jObject = new JSONObject(page);
			JSONArray jArray = jObject.getJSONArray("items");
			for (int i = 0; i < jArray.length(); i++) {
				if (!jArray.isNull(i)) {
					JSONObject item = jArray.getJSONObject(i).getJSONObject("product");
				    String title = item.getString("title");
				    String brand = item.getString("brand");
				    Log.d(TAG, "Title=" + title + ", brand=" + brand);
				    result = title;
				    break;
				    //TODO: return all values in a list;
				}
			}
		}
		catch (Exception e) {
			Log.e(TAG, "Error parsing page");
		}
		return result;
	}
}
