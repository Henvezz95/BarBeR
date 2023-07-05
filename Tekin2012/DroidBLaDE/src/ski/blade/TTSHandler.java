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

import java.util.ArrayList;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.speech.tts.TextToSpeech;
import android.speech.tts.TextToSpeech.OnInitListener;
import android.speech.tts.TextToSpeech.OnUtteranceCompletedListener;
import android.util.Log;

/**
 * @class TTS handler class
 * @author kamyon
 * Verifies that TTS is installed, and if it is, is used to control the TTS behavior.
 */
public class TTSHandler implements OnInitListener, OnUtteranceCompletedListener{
	private TextToSpeech mTTS;
	private Context mContext;
	
	private final String TAG = "BLaDE TTSHandler";

	/**
	 * Constructor
	 * @param context main context the TTS instance will be running in
	 */
	public TTSHandler(Context context) {
		mContext = context;
		mTTS = new TextToSpeech(mContext, this);
	}
	
	/**
	 * Verifies that TTS is installed on the device.
	 * @param activity Activity that will receive a callback at onActivityResult() with the TTS check result.
	 * @param reqCode code to return when the TTS verification activity returns.
	 */
	static public void verifyTTS(Activity activity, int reqCode) {
		//Check to make sure that TTS exists:
		Intent verifyTTS = new Intent();
		
		ArrayList<String> checkVoiceDataFor = new ArrayList<String>(1);
		checkVoiceDataFor.add("eng");   // installed on emulator
		verifyTTS.setAction(TextToSpeech.Engine.ACTION_CHECK_TTS_DATA);
		verifyTTS.putStringArrayListExtra(TextToSpeech.Engine.EXTRA_CHECK_VOICE_DATA_FOR, checkVoiceDataFor);
		activity.startActivityForResult(verifyTTS, reqCode);		
	}

	@Override
	public void onInit(int status) {
		Log.v(TAG, "TTS initialized");
		if (status == TextToSpeech.SUCCESS) {
			//speak(mContext.getText(R.string.hello).toString());
			speak(R.string.hello);
			mTTS.setOnUtteranceCompletedListener(this);
		}
		else
			Log.e(TAG, "Problem initializing TTS");
	}
	
	/**
	 * Stops the tts.
	 */
	public void stop() {
		if (mTTS != null)
			mTTS.stop();
		else
			Log.d(TAG, "TTS not initialized");
	}
	
	/**
	 * Releases TTS resources.
	 */
	public void release() {
		if (mTTS != null) {
			mTTS.shutdown();
			mTTS = null;
		}
		else
			Log.d(TAG, "TTS not initialized or already released");
		mContext = null;
	}
	
	/**
	 * Speaks out loud a given text
	 * @param aText text to speak out
	 */
	public void speak(String aText) {
		if (mTTS != null)
			mTTS.speak(aText, TextToSpeech.QUEUE_ADD, null);
		else
			Log.e(TAG, "TTS not initialized");
	}
		
	/**
	 * Speaks out loud a given resource
	 * @param res resource id that should correspond to a text to be spoken out
	 */
	public void speak(int res) {
		speak(mContext.getText(res).toString());
	}

	/**
	 * Speaks out loud a given text immediately, cutting off another utterance if necessary
	 * @param aText text to speak out
	 */
	public void speakNow(String aText) {
		if (mTTS != null)
		{
			mTTS.stop();
			mTTS.speak(aText, TextToSpeech.QUEUE_ADD, null);
		}
		else
			Log.e(TAG, "TTS not initialized");
	}
		
	/**
	 * Speaks out loud a given resource, cutting off another utterance if necessary
	 * @param res resource id that should correspond to a text to be spoken out
	 */
	public void speakNow(int res) {
		speakNow(mContext.getText(res).toString());
	}

	@Override
	public void onUtteranceCompleted(String utteranceId) {
		Log.d(TAG, "Just finished saying " + utteranceId);
	}
	
	protected void finalize() {
		release();
	}

}
