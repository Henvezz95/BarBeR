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

import android.os.AsyncTask;
import android.util.Log;

public class UPCLookup {
	
	private OnWebLookup callback_;
	private LookupImplementation lookupImplementation_;
	private WebLookup weblookup_; 
	
	private final String TAG = "BLaDE UPCLookup";
	
	public UPCLookup(OnWebLookup aLookupCallback, WebLookup aWebLookup) {
		callback_ = aLookupCallback;
		weblookup_ = aWebLookup;
	}

	private class LookupImplementation extends AsyncTask<String, Void, String>{
		@Override
		protected String doInBackground(String... params) {
			String upc = params[0], result = null;
			try {
				result = weblookup_.performLookup(upc);
			}
			catch (Exception e) {
				e.printStackTrace();
			}
			return result;
		}
		
		@Override
		protected void onPreExecute() {
			callback_.indicateProgress(0);
		}
		
		@Override
		protected void onPostExecute(String result) {
			callback_.indicateProgress(100);
			if (result != null)
				Log.d(TAG, "UPC Lookup result: " + result);
			else
				Log.d(TAG, "Product info not found");
			callback_.onProductInfoReceived(result);
		}
		
		@Override
		protected void onCancelled() {
			Log.d(TAG, "UPC Lookup canceled");
			weblookup_.cancel();
		}
	}
	
	/**
	 * Performs a upc->product lookup
	 * @param upc upc code to look up.
	 * @throws Exception if the product information cannot be obtained.
	 */
	public void performLookup(String upc) throws Exception {
		try {
			lookupImplementation_ = new LookupImplementation();
			lookupImplementation_.execute(upc);
		}
		catch (Exception e)	{
			e.printStackTrace();
		}
	}
	
	/**
	 * Cancels an ongoing lookup
	 */
	public void cancelLookup() {
		if (lookupImplementation_ != null)
		{
			callback_.indicateProgress(-1);
			lookupImplementation_.cancel(true);
			lookupImplementation_ = null;
		}
	}

	public void release() {
		cancelLookup();
		callback_ = null;
	}
	
	@Override
	protected void finalize() {
		release();
	}

	interface OnWebLookup {
		/**
		 * Callback to indicate progress
		 * @param progress percentage [0,...,100] of progress made
		 */
		void indicateProgress(int progress);
		
		/**
		 * Callback when product information has been received
		 * @param result product information to publish
		 */
		void onProductInfoReceived(String result);
	}
	
	interface WebLookup {
		/**
		 * Performs a web lookup for barcode
		 * @param upc barcode string to search for
		 * @return product information for this barcode
		 */
		public String performLookup(String upc);
		
		/**
		 * Cancels an ongoing lookup
		 */
		public void cancel();
	}
}
