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

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.GestureDetector;
import android.view.GestureDetector.OnGestureListener;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.MotionEvent;
//import android.view.View;
import android.widget.TextView;

//TODO: See if we can just use the opencv headers without having to link to the library
public class BLaDEActivity extends Activity {
	static {
		System.loadLibrary("BLaDE");
		System.loadLibrary("BLaDE_JNI");
	}
	
	/** Expected width and height of preview: TODO: move to CameraHandler_ and make sure it is supported */
	private final int width = 640, height = 480;
	/** Class that processed each video frame */
	private FrameProcessor mProc;
	/**Provides input/output to the user */
	private UserInterfaceProvider uiProvider_;
	
	/** Scans for gestures */
	private GestureHandler gestureHandler_;
	/** Handles barcode detection results */
	private BarcodeResultHandler barcodeHandler_ = new BarcodeResultHandler();
	
	private final String TAG = "BLaDE Activity";

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

		gestureHandler_ = new GestureHandler(this); //TODO: move GestureDetector to uiProvider if necessary to keep it.
		if (gestureHandler_ == null)
			Log.e(TAG, "Gesture handler is null!");

		//Create new frame processor TODO: change so that we only pass this, from which the frame processor can retrieve mPreviewSV and mOverlaySV.
		mProc = new FrameProcessor(width, height, barcodeHandler_, this);
		
		//Initialize the UI Provider
		uiProvider_ = new UserInterfaceProvider(this);
		
    }
    
    @Override
    public void onStart() {
    	super.onStart();
    	uiProvider_.start((TextView) findViewById(R.id.textInfo));
    }
    
    @Override
    public void onPause() {
    	super.onPause();
    	if (uiProvider_ != null)
    		uiProvider_.stop();
    }
    
    @Override
    public void onDestroy() {
    	super.onDestroy();
    	exit();
    }
    
    @Override
	protected void onActivityResult(int reqCode, int resCode, Intent intent) {
		if (reqCode == UserInterfaceProvider.VERIFY_TTS) { 	//Result of TTS check - installs TTS if needed
			switch (resCode) {
			case TextToSpeech.Engine.CHECK_VOICE_DATA_PASS:
				//TTS is available
				uiProvider_.setTtsVerified(true);
				break;
			case TextToSpeech.Engine.CHECK_VOICE_DATA_BAD_DATA:
			case TextToSpeech.Engine.CHECK_VOICE_DATA_MISSING_DATA:
			case TextToSpeech.Engine.CHECK_VOICE_DATA_MISSING_VOLUME:
				Intent installTTS = new Intent();
				installTTS.setAction(TextToSpeech.Engine.ACTION_INSTALL_TTS_DATA);
				startActivity(installTTS);
				break;
			case TextToSpeech.Engine.CHECK_VOICE_DATA_FAIL:
				Log.e(TAG, "TTS not available.");
				exit();
			}
		}
	}
	
    /**
     * Exits program
     */
	public void exit() {
		if (mProc != null) {
			mProc.release();
			mProc = null;
		}
    	if (uiProvider_ != null) {
    		uiProvider_.release();
    		uiProvider_ = null;
    	}
    	if (barcodeHandler_ != null) {
    		barcodeHandler_.release();
    		barcodeHandler_ = null;
    	}
    	if (gestureHandler_ != null) {
    		gestureHandler_.release();
    		gestureHandler_ = null;
    	}
		finish();
	}
	
	/**
	 * @class Handles barcode detection and lookup callbacks
	 * @author kamyon
	 *
	 */
	private class BarcodeResultHandler implements FrameProcessor.OnBarcodeFound, UPCLookup.OnWebLookup {
		/** Performs UPC Lookup */
		private UPCLookup lookup_ = null;

		//Callbacks from UPCLookup 
		@Override
		public void onProductInfoReceived(String result) {
			if (result == null)
				result = "Product Not Found";
			try {
				uiProvider_.stopWaitingIndication();
				//((TextView) findViewById(R.id.textInfo)).setText(result);
				uiProvider_.output(result);
				Log.d(TAG, "Product" + result);
			} 
			catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		@Override
		public void indicateProgress(int progress) {
			if (uiProvider_ != null)
			{
				if (progress <= 0)
					uiProvider_.indicateWaiting();
				else if (progress == 100)
					uiProvider_.stopWaitingIndication();
				else
					Log.e(TAG, "Should not be indicating progress other than -1, 0 or 100");
			}
		}
	
		//Callbacks from FrameProcessor
		@Override
		public void onBarcodeDetected(Barcode bc) {
			Log.d(TAG, "Barcode detected");
		}
		
		@Override
		public void onBarcodeDecodingFailed(Barcode bc) {
			Log.d(TAG, "Barcode decoding attempt failed");
		}
	
		@Override
		public void onBarcodeDecoded(Barcode bc) {
			Log.d(TAG, "Barcode decoded");
			//Display found barcode
			//((TextView) findViewById(R.id.textUPC)).setText(bc.UPC);
			//Pause processing
			mProc.pause();
			//Lookup UPC info
	    	Log.d(TAG, "Looking up " + bc.UPC);
			((TextView) findViewById(R.id.textUPC)).setText(bc.UPC);
			//Do the lookup if requested
			if (lookup_ != null)
			{
				uiProvider_.output(R.string.barcode_lookup);
				try {
					lookup_.performLookup(bc.UPC);
				}
				catch (Exception e) {
					e.printStackTrace();
				}
			}
			else
			{
				uiProvider_.output(R.string.barcode_found);
				uiProvider_.outputAdditional(bc.UPC);
			}
		}
		
		/**
		 * Activates web lookup if not already active
		 */
		public void activateWebLookup() {
	    	if (lookup_ == null)
	    		lookup_ = new UPCLookup(this, new GooglePSLookup());
	    	else
	    		Log.d(TAG, "Web lookup already active");
		}
		
		/**
		 * Deactivates web lookup if active
		 */
		public void deactivateWebLookup() {
    		if (lookup_ != null) {
    			lookup_.release();
        		lookup_ = null;
    		}
    		else
    			Log.d(TAG, "Web lookup already deactivated");
		}
		
		/**
		 * Checks whether web lookup is active
		 * @return true if lookup is active, false if not.
		 */
		public boolean isWebLookupActive() {
			return (lookup_ != null);
		}
		
		/**
		 * Releases any resources used by this class
		 */
		public void release() {
	    	if (lookup_ != null) {
	    		lookup_.release();
	    		lookup_ = null;
	    	}
		}
		
		@Override
		protected void finalize() {
			release();
		}
	}
		
	//Options Menu
	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
	    MenuInflater inflater = getMenuInflater();
	    inflater.inflate(R.menu.options_menu, menu);	//TODO populate menu from last saved state
	    return true;
	}
		
	@Override
	public boolean onOptionsItemSelected(MenuItem item) {
		FrameProcessor.ProcessorOptions opts = mProc.getOptions();
		if (opts == null) {
			Log.e(TAG, "Options is NULL!");
			return false;
		}
	    // Handle item selection
	    switch (item.getItemId()) {
	    case R.id.audio_fb:
	    	//Switch feedback
			opts = mProc.getOptions();
	    	opts.isAudioFeedbackOn = !opts.isAudioFeedbackOn;
	    	mProc.setOptions(opts);
	    	opts = mProc.getOptions(); //In case there is a problem changing feedback
	    	item.setTitle(opts.isAudioFeedbackOn ? R.string.audio_fb_off : R.string.audio_fb_on);
	        Log.d(TAG, "Audio feedback " + (opts.isAudioFeedbackOn ? "on" : "off") );
	        return true;
	    case R.id.web_lookup:
	    	if (barcodeHandler_.isWebLookupActive()) {
	    		barcodeHandler_.deactivateWebLookup();
	    		item.setTitle(R.string.web_lookup_on);
	    	}
	    	else {
	    		barcodeHandler_.activateWebLookup();
	    		item.setTitle(R.string.web_lookup_off);
	    	}
	        Log.d(TAG, "Web lookup" + (barcodeHandler_.isWebLookupActive() ? "on" : "off") );
	        return true;
	    case R.id.bc_decoding:
	    	//Turn on/off barcode decoding step:
			opts = mProc.getOptions();
			opts.isBarcodeDecodingOn = !opts.isBarcodeDecodingOn;
			mProc.setOptions(opts);
	    	opts = mProc.getOptions(); //In case there is a problem changing feedback
	    	item.setTitle(opts.isBarcodeDecodingOn ? R.string.bc_decoding_off : R.string.bc_decoding_on);
	        Log.d(TAG, "Barcode decoding " + (opts.isBarcodeDecodingOn ? "on" : "off") );
	    	return true;
	    case R.id.horiz_fb:
	    	//Turn on/off vibration feedback to indicate horizontal alignment:
			opts = mProc.getOptions();
			opts.isTiltSensingOn = !opts.isTiltSensingOn;
			mProc.setOptions(opts);
	    	opts = mProc.getOptions(); //In case there is a problem changing feedback
	    	item.setTitle(opts.isTiltSensingOn ? R.string.horiz_fb_off : R.string.horiz_fb_on);
	        Log.d(TAG, "Tilt sensing " + (opts.isTiltSensingOn ? "on" : "off") );
	    	return true;
	    default:
	        return super.onOptionsItemSelected(item);
	    }
	}
	
	/**
	 * @class Callbacks from gesturedetector
	 * TODO: maybe move to uiprovider?
	 * @author kamyon
	 *
	 */
	private class GestureHandler implements OnGestureListener {
		/** Gesture scanner */
		private GestureDetector gestureScanner_;

		/** Used to display barcode scan line, etc. */
		private OverlayView overlaySV_;

		public GestureHandler(Activity act) {
			overlaySV_ = (OverlayView) act.findViewById(R.id.overlay);
			gestureScanner_ = new GestureDetector(act.getApplicationContext(), this); //TODO: move GestureDetector to uiProvider if necessary to keep it.
			overlaySV_.setGestureListener(gestureScanner_);
		}
		@Override
		public boolean onDown(MotionEvent e) {
			// TODO Auto-generated method stub
			return true;
		}
	
		@Override
		public boolean onFling(MotionEvent e1, MotionEvent e2, float velocityX, float velocityY) {
			// TODO Auto-generated method stub
			return false;
		}
	
		@Override
		public void onLongPress(MotionEvent e) {
			// TODO Auto-generated method stub
			Log.d(TAG, "Long press caught");
		}
	
		@Override
		public boolean onScroll(MotionEvent e1, MotionEvent e2, float distanceX, float distanceY) {
			//Log.d(TAG, "Changing sound feedback status to" + (isSoundFeedbackOn ? "off" : "on"));
			//isSoundFeedbackOn = !isSoundFeedbackOn;
			return false;
		}
	
		@Override
		public void onShowPress(MotionEvent e) {
			// TODO Auto-generated method stub
		}
	
		@Override
		public boolean onSingleTapUp(MotionEvent e) {
			// TODO Auto-generated method stub
			Log.d(TAG, "Caught surface tap");
			if (!mProc.isActive())
			{
				mProc.start();
				return true;
			}
			else
				return false;
		}
		
		/**
		 * Releases the gesture handler
		 */
		public void release() {
			if (overlaySV_ != null)
			{
				overlaySV_.setGestureListener(null);
				overlaySV_ = null;
				gestureScanner_ = null;
			}
		}
		
		@Override
		protected void finalize() {
			release();
		}
	}
}