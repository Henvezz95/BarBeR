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

import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;
import android.util.Log;

/*
 * TODO: maybe we can move the actual write operations etc. to a new thread???
 * See http://stackoverflow.com/questions/4425526/android-mixing-multiple-audiotrack-instances
 * This way, we may be able play the the background click on a separate thread at a lower rate as well.
 */
public class SoundFeedback {
	/** Audio playback handler */
	private AudioTrack track;
	/** sampling rate of audio buffer*/
	private final int rate = 16000;
	/** Size of buffer in number of samples. Each buffer will be 2 * #channels * nSamplesInBuffer bytes long*/
	private int nSamplesInBuffer = 800;
	/** Frequency of audio tone to use */
	private final int frequency = 1000;
	/** Prefilled audio buffers for the different feedback levels */
	private short[][][] buffer;
	/** Null buffer - "not crashed" background sound */
	private short [] nullBuffer;
	/** Empty stream initially written to the playback buffer - TODO: why not write the null buffer?*/
	private short [] emptyBuffer;
	/** alignment feedback value to provide */
	private volatile int alignment;
	/** distance feedback value to provide */ 
	private volatile int distance;
	/** Whether feedback is on or off */
	private boolean isFeedbackOn_ = true;
	/** Audio feedback status */
	public static final int UNINITIALIZED = AudioTrack.STATE_UNINITIALIZED;
	public static final int STOPPED = AudioTrack.PLAYSTATE_STOPPED;
	public static final int PAUSED = AudioTrack.PLAYSTATE_PAUSED;
	public static final int PLAYING = AudioTrack.PLAYSTATE_PLAYING;
	/** Periodic callback handler */
	private AudioCallbackHandler callback_;
	
	private final String TAG = "BLaDE SoundFeedback";
	
	/**
	 * Constructor
	 */
	public SoundFeedback() {
		//Allocate the audio buffer
		//get minimum possible size for the given format
		int minSize = AudioTrack.getMinBufferSize(rate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT);
		//allocate enough to hold about 2 stream buffers to ensure continuous playback
		if (minSize > 4 * nSamplesInBuffer)
			nSamplesInBuffer = minSize / 4 + (minSize % 4 > 0 ? 1 : 0);
		int allocatedBufferSize = 4 * nSamplesInBuffer;	//in bytes
		emptyBuffer = new short[nSamplesInBuffer];
		//Prefill audio stream buffers */
		createStreams(nSamplesInBuffer);
		track = new AudioTrack(AudioManager.STREAM_MUSIC, rate, AudioFormat.CHANNEL_OUT_MONO, 
				AudioFormat.ENCODING_PCM_16BIT, allocatedBufferSize, AudioTrack.MODE_STREAM);
		//Get notification every time a stream is fully played
		callback_ = new AudioCallbackHandler();
		track.setPositionNotificationPeriod(nSamplesInBuffer);
		track.setPlaybackPositionUpdateListener(callback_);
		//TODO: see if we should adjust volume levels similar to the soundresourcehandler
		track.setStereoVolume(AudioTrack.getMaxVolume(), AudioTrack.getMaxVolume());
	}
	
	private void play(int alignment, int distance) {
		Log.d(TAG, "Sending sound feedback for Distance = " + distance + ", Alignment = " + alignment);
		write(buffer[distance][alignment]);
	}
	
	/**
	 * Starts the audio feedback
	 */
	public void start() {
		if (getState() != AudioTrack.PLAYSTATE_PLAYING) {
			//Fill buffer
			write(emptyBuffer);
			write(emptyBuffer);
			track.play();
			Log.d(TAG, "Starting audio feedback...Status=" + getState());
		}
	}
	
	/**
	 * Plays a silent buffer or the background "click" to designate that application is still active.
	 * @param isSilent if true, no sound is played, 0 is written to the playback buffer. 
	 * If false, the background click is played.
	 */
	public void nullPlay(boolean isSilent) {
		write((isSilent ? emptyBuffer : nullBuffer));
	}
	
	/**
	 * Pauses the audio feedback
	 */
	public void pause() {
		if (getState() == PLAYING) {
			//track.flush();
			track.pause();
			Log.d(TAG, "Pausing audio feedback...Status=" + getState());
		}
	}
	
	/**
	 * Stops the audio feedback
	 */
	public void stop() {
		if ( (getState() == PLAYING) || (getState() == PAUSED) ){
			track.stop();
			track.flush();
		}
	}	

	/**
	 * Releases the sound feedback resources
	 */
	public void release() {
		//Stop playback
		stop();
		//Release resources
		if (track != null) {
			track.release();
			track = null;
		}
		if (callback_ != null)
			callback_ = null;
	}
	
	/**
	 * Gets the state of the current audio track
	 * @return audio track state (playing, paused or stopped)
	 */
	public int getState() {
		if (track != null)
			return track.getPlayState();
		else
			return AudioTrack.STATE_UNINITIALIZED;
	}

	/**
	 * Changes the stream that will be added to the playback buffer next time.
	 * @param bc barcode to calculate feedback for
	 * @param w width of the frame
	 * @param h height of the frame
	 */
	public void set(Barcode bc, int w, int h) {
		if (bc == null) {
			alignment = distance = 0;
		}
		else {
			//Calculate alignment
			Log.d(TAG, "Barcode found between (" + bc.x1 + "," + bc.y1 + ")-(" + bc.x2 + "," + bc.y2 + ") in frame of size " + w + "x" + h);
			int minDist = Math.min(w, h) / 20; //how far the edges should be from the edge of the image
			double dX = bc.x2 - bc.x1, dY = bc.y2 - bc.y1;
			double angle = Math.atan(dY / dX);
			double bcWidth = Math.sqrt(dX * dX + dY * dY);
			double maxWidth = .8 * Math.min(w / Math.cos(angle), h / Math.abs(Math.sin(angle)) ), minWidth = 0.5 * maxWidth;
			Log.d(TAG, "Barcode angle: " + angle + ", wMax = " + maxWidth + ", wMin = " + minWidth + ", w = " + bcWidth);
			int leftDist = Math.min(bc.x1, bc.x2), rightDist = w - Math.max(bc.x1, bc.x2);
			int topDist = Math.min(bc.y1, bc.y2), botDist = h - Math.max(bc.y1, bc.y2);
			//Now calculate the scores
			alignment = distance = 5;
			if ( bcWidth > maxWidth )
				distance *= maxWidth / bcWidth;
			else if (bcWidth < minWidth)
				distance *= bcWidth / minWidth;
			double alignmentScale = 1.0;
			if (leftDist < minDist)
				alignmentScale *= .5 * leftDist / minDist + .5;
			if (rightDist < minDist)
				alignmentScale *= .5 * rightDist / minDist + .5;
			if (topDist < minDist)
				alignmentScale *= .5 * topDist / minDist + .5;
			if (botDist < minDist)
				alignmentScale *= .5 * botDist / minDist + .5;
			alignment *= alignmentScale;
			if (alignment < 0) {
				alignment = 0;
				Log.e(TAG, "Alignment is negatve!! WTF?");
			}
		}
	}
	
	/**
	 * Write an audio stream to the playback buffer 
	 * @param buf buffer to write
	 */
	private void write(short[] buf) {
		if (track != null) {
			int written = track.write(buf, 0, nSamplesInBuffer);
			if (written == AudioTrack.ERROR_BAD_VALUE)
				Log.e(TAG, "Feedback: Bad value!");
			else if (written == AudioTrack.ERROR_INVALID_OPERATION)
				Log.e(TAG, "Feedback: Invalid Operation!");
		}
		else
			Log.d(TAG, "Feedback: trying to write to uninitialized AudioTrack");
	}
	
	/**
	 * Creates the streams that will be written to the playback suffer for various feedback values
	 * Called during initialization.
	 * @param nSamplesInBuffer number of samples that will be created in the stream buffers
	 */
	private void createStreams(int nSamplesInBuffer){
		buffer = new short[6][6][nSamplesInBuffer];
		double w = 2 * Math.PI * frequency / rate, volume;
		int dutyCycle;
		//We create buffers for different volumes, since otherwise volume adjustment does not seem to be fast enough. 
		for (int d = 0; d <= 5; d++) {
			volume = (Short.MAX_VALUE >> (5-d)); 
			for (int a = 0; a <= 5; a++) {
				dutyCycle = a * nSamplesInBuffer / 5;
				for (int t = 0; t < dutyCycle; t++)
					buffer[d][a][t] = (short) (volume * Math.cos(w * t));
				for (int t = dutyCycle; t < nSamplesInBuffer; t++)
					buffer[d][a][t] = 0;
			}
		}
		volume = Short.MAX_VALUE / 5;
		dutyCycle = nSamplesInBuffer / 20;
		nullBuffer = new short[nSamplesInBuffer];
		for (int t = 0; t < dutyCycle; t++)
			nullBuffer[t] =  (short) (volume * Math.cos(w * t));
		for (int t = dutyCycle; t < nSamplesInBuffer; t++)
			nullBuffer[t] = 0;
	}
	
	/**
	 * Checks whether audio feedback is turned on
	 * @return true if audio barcode feedback is on
	 */
	public boolean isFeedbackOn() {
		return isFeedbackOn_;
	}
	
	/**
	 * Turns feedback on/off
	 * @param isFeedbackOn if true, feedback is turned on. If false, it is turned off
	 */
	public void setFeedback(boolean isFeedbackOn) {
		isFeedbackOn_ = isFeedbackOn;
        Log.d(TAG, "Audio feedback turned " + (isFeedbackOn ? "on" : "off"));
	}
	
	@Override
	protected void finalize() {
		release();
	}
	
	/**
	 * @class Class that handler audio callbacks
	 * @author kamyon
	 *
	 */
	private class AudioCallbackHandler implements AudioTrack.OnPlaybackPositionUpdateListener {
		private static final int NULL_SOUND_PERIOD = 20;
		private int callbackCount_ = 0;
		@Override
		public void onMarkerReached(AudioTrack track) {
			//We do not use markers, so no implementation here
		}

		@Override
		public void onPeriodicNotification(AudioTrack track) {
			//Increase callback count
			callbackCount_++;
			if (callbackCount_ >= NULL_SOUND_PERIOD)
				callbackCount_ = 0;
			//We get periodic callbacks here - see if there is a barcode detected and assign appropriate feedback.
			if ((alignment == 0) || (distance == 0) || !isFeedbackOn_)
				nullPlay(callbackCount_ > 0);
			else
				play(alignment, distance);
		}		
	}
}
