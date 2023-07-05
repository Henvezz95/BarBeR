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
import android.content.Context;
import android.widget.TextView;

/**
 * @class Class to provide an accessible interface in the form of TTS or text views to be read by a screen reader
 * takes care of the appropriate method of providing information and feedback to the user.
 * TODO: TTS is working, but the text views need to be improved and tested with screenreaders.
 * @author kamyon
 *
 */
public class UserInterfaceProvider {

	/** Sound handler */
	private SoundResourceHandler soundResourceHandler_ = null;
	/** Whether usingTTS directly or assuming a screen reader is installed */
	private boolean isUsingTts_ = true; //TODO: change to false after testing to set the screenreader mode as default
	/** TTS Handler */
	private TTSHandler ttsHandler_ = null;
	/** Result to expect from TTS Verification Activity */
	static public final int VERIFY_TTS = 0;
	/** Application context */
	private Context appContext_ = null;
	/** Sound resource to be used to indicate that we are waiting for an action to complete */
	private final int WAITING_SOUND_RESOURCE = R.raw.clock;
	/** Waiting sound id to use */
	private int waitingSoundId_;
	/** Text display to use */
	private TextView textView_ = null;
	
	private final String TAG = "BLaDE UIProvider";

	/**
	 * Constructor
	 * @param act parent activity
	 */
	public UserInterfaceProvider(Activity act) {
		//Get the application context
		appContext_ = act.getApplicationContext();
		
		//Create the sound handler - will play sound resources when needed
		soundResourceHandler_ = new SoundResourceHandler(act);
		//Load the sound to be played during web lookup
		waitingSoundId_ = soundResourceHandler_.load(WAITING_SOUND_RESOURCE);
		
		//Check if TTS is available. TTS will be initialized if verification passes
		TTSHandler.verifyTTS(act, VERIFY_TTS);
	}
	
	/**
	 * Informs whether TTS is installed on the syste,
	 * @param isVerified true if TTS has been determined to be installed, false otherwise.
	 */
	public void setTtsVerified(boolean isVerified) {
		//Create the tts handler if verified - ignore tts otherwise and just display results
		if (isVerified)
			ttsHandler_ = new TTSHandler(appContext_);
		else
			isUsingTts_ = false;
	}

	/**
	 * Based on the tts mode, either displays a given text in a text view, or uses tts to speak it out.
	 * @param aText text to provide
	 */
	public void output(String aText) {
		if (isUsingTts_)
			ttsHandler_.speakNow(aText);
		else
			display(aText);
	}
	
	/**
	 * Based on the tts mode, either displays a given text resource in a text view, or uses tts to speak out the resource text.
	 * @param resId id of a text resource
	 */
	public void output(int resId) {
		output(appContext_.getString(resId));
	}
	
	/**
	 * method to add data that is delayed or appended without cutting off the previous data
	 * @param resId resource id of the new data to be added
	 */
	public void outputAdditional(String aText) {
		if (isUsingTts_)
			ttsHandler_.speak(aText);
		else
			displayAdditional(aText);
	}

	/**
	 * method to add data that is delayed or appended without cutting off the previous data
	 * @param resId resource id of the new data to be added
	 */
	public void outputAdditional(int resId) {
		outputAdditional(appContext_.getString(resId));
	}
	
	/**
	 * Starts after a reinitialization. Should be called every time the parent activity is restarted to refresh the text view.
	 * @param aView
	 */
	public void start(TextView aView) {
		textView_ = aView;
	}
	
	/**
	 * Displays a given text in a text view for a screen reader to read out loud
	 * @param aText text to display.
	 */
	private void display(String aText) {
		if (textView_ != null)
			textView_.setText(aText);
	}
	
	/**
	 * Display additional text on the existing text view if one is visible
	 * @param aText additional text to display
	 */
	private void displayAdditional(String aText) {
		if (textView_ != null)
			textView_.append(aText);
	}
	
	/**
	 * Indicate that we are waiting for an action to complete
	 */
	public void indicateWaiting() {
		if (soundResourceHandler_ != null)
			soundResourceHandler_.play(waitingSoundId_, true);
	}
	
	/**
	 * Stops the waiting indication
	 */
	public void stopWaitingIndication() {
		if (soundResourceHandler_ != null)
			soundResourceHandler_.stop(waitingSoundId_);		
	}
	
	/**
	 * Stops the provided feedback temporarily
	 */
	public void stop() {
		if (ttsHandler_ != null)
			ttsHandler_.stop();
		if (soundResourceHandler_ != null)
			soundResourceHandler_.stop();
	}

	/**
	 * Releases the user interface resources
	 */
	public void release() {
		if (soundResourceHandler_ != null) {
			soundResourceHandler_.release();
			soundResourceHandler_ = null;
		}
		if (ttsHandler_ != null) {
			ttsHandler_.stop();
			ttsHandler_.release();
			ttsHandler_ = null;
		}
		appContext_ = null;
	}
	
}
