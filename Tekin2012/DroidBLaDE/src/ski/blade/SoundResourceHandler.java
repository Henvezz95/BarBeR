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
import java.util.List;

import android.app.Activity;
import android.content.Context;
import android.media.AudioManager;
import android.media.SoundPool;
import android.util.Log;

public class SoundResourceHandler  implements SoundPool.OnLoadCompleteListener {
	private SoundPool snds_;
	private float volume_;
	private List<Integer> soundIds_ = new ArrayList<Integer>();
	private Context appContext_;
	private int currentlyPlayingSoundId_ = 0;
	private int streamId_ = 0;
	
	private final String TAG = "BLaDE SoundResourceHandler";

	/**
	 * Handles loading/unloading and playing of sound resources.
	 * @param act this activity
	 */
	public SoundResourceHandler(Activity act) {
		//Get application context
		appContext_ = act.getApplicationContext();
        //Create sound pool
    	snds_ = new SoundPool(2, AudioManager.STREAM_MUSIC, 0);
    	snds_.setOnLoadCompleteListener(this);
    	// Getting the user sound settings
		AudioManager audioManager = (AudioManager) act.getSystemService(Context.AUDIO_SERVICE);
		float actualVolume = (float) audioManager.getStreamVolume(AudioManager.STREAM_MUSIC);
		float maxVolume = (float) audioManager.getStreamMaxVolume(AudioManager.STREAM_MUSIC);
		volume_ = 0.5f * actualVolume / maxVolume;
	}
	
	/**
	 * Loads a resource into the sound pool, from where it can be played.
	 * @param resId resource id to load
	 * @return a soundId which can be used to play or unload the sound pool.
	 */
	public int load(int resId) {
		return snds_.load(appContext_, resId, 1);
	}
	
	/**
	 * Unloads a sound from the pool
	 * @param sndId sound id to unload
	 * @return true if sound is successfully unloaded, false if it was already unloaded or did not exists in the sound pool.
	 */
	public boolean unload(int sndId) {
		//Stop if this is the current sound playing
		if (sndId == currentlyPlayingSoundId_)
			stop(sndId);
		//Remove from list of available sounds
		soundIds_.remove(Integer.valueOf(sndId));
		return snds_.unload(sndId);
	}

	@Override
	public void onLoadComplete(SoundPool soundPool, int sampleId, int status) {
		//If successfully loaded, add to the list of available sounds
		if (status == 0)
			soundIds_.add(sampleId);
	}
	
	/**
	 * Plays the sound, will stop a previously playing sound.
	 * @param sndId id of the sound to play
	 * @param isLooping true if the sound is to be played in a looping fashion. false to play once
	 * @return true if sound is successfully played, false otherwise. 
	 */
	public boolean play(int sndId, boolean isLooping){
		if (isAvailable(sndId)) {
			if (isPlaying()) {
				Log.w(TAG, "Play requested. Stopping stream " + currentlyPlayingSoundId_);
				stop();
			}
			streamId_ = snds_.play(sndId, volume_, volume_, 1, (isLooping ? -1 : 0), 1);
			if (streamId_ != 0) {
				currentlyPlayingSoundId_ = sndId;
				return true;
			}
			else
				return false;
		}
		else
		{
			Log.e(TAG, "Sound resource " + sndId + "not yet loaded");
			return false;
		}
	}

	/**
	 * Returns whether a given sound is playing
	 * @param sndId a sound id
	 * @return true if sndId is currently playing.
	 */
	public boolean isPlaying(int sndId) {
		return ( (sndId != 0) && (currentlyPlayingSoundId_ == sndId) );
	}
	
	/**
	 * Returns whether a sound is playing
	 * @return true if a sound is currently playing.
	 */
	public boolean isPlaying() {
		return isPlaying(currentlyPlayingSoundId_);
	}
	
	/**
	 * Stops the requested sound if it is currently playing
	 * @param sndId id of the sound to stop
	 * @return false if the requested sound was not playing.
	 */
	public boolean stop(int sndId) {
		if ( (sndId == currentlyPlayingSoundId_) && (sndId != 0) ){
			snds_.stop(streamId_);
			currentlyPlayingSoundId_ = 0;
			return true;
		}
		else
			return false;
	}

	/**
	 * Stops currently playing sound
	 * @return false if no sound was currently playing.
	 */
	public boolean stop() {
		return stop(currentlyPlayingSoundId_);
	}
	
	/**
	 * Checks whether a sound id is available for playback
	 * @param sndId id of the sound to check for
	 * @return true if the sound can be played. false if it is not loaded.
	 */
	public boolean isAvailable(int sndId) {
		return ((sndId != 0) && soundIds_.contains(sndId));
	}

	void release() {
		//unload all sounds
		for (int sndId: soundIds_)
			unload(sndId);
		//release the sound container
		if (snds_ != null)
		{
			snds_.release();
			snds_ = null;
		}
		//release application context
		appContext_ = null;
	}
	
	protected void finalize() {
		release();
	}
}
