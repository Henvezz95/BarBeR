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

/*
 * SoundManager.h
 *
 *  Created on: Jul 30, 2012
 *      Author: kamyon
 */

#ifndef SOUNDMANAGER_H_
#define SOUNDMANAGER_H_

#include "ski/types.h"
#include <string>
#include <memory>

//Forward declaration of implementation
class AlsaAccess;

/**
 * @class Sound manager class
 */
class SoundManager
{
public:

	///Audio data is limited to signed 16-bit int.
	typedef TInt16 TAudioData;

	/** Audio parameters to use. For simplicity, we assume the device supports interleaved signed 16-bit samples
	 * Thus, for a 2 channel signal, the audio frame byte-pattern should look like:
	 * L1 L1 R1 R1 L2 L2 R2 R2 ...
	 * See http://www.alsa-project.org/main/index.php/FramesPeriods for more details
	 */
	struct Parameters
	{
		/** Number of channels */
		TUInt nChannels;
		/** Sampling rate */
		TUInt samplingRate;
		/** Period size in frames = # of frames per period*/
		TUInt64 periodSize;
		/** Number of periods the buffer will contain */
		TUInt nPeriods;
		/** Buffer size in number of frames - this will be auto-set by the hardware*/
		TUInt64 bufferSize;
		/**
		 * Default constructor
		 * @param nC number of channels
		 * @param aRate sampling rate
		 * @param aPerSz number of frames per period
		 * @param nP number of periods in buffer
		 */
		Parameters(TUInt nC = 2, TUInt aRate=16000, TUInt64 aPerSz=4096, TUInt nP=4);
		/** Frame size in bytes */
		TUInt frameSizeInBytes() const;
		/** Period size in bytes */
		TUInt64 periodSizeInBytes() const;
	};

	enum Status
	{
		STATUS_CLOSED,
		STATUS_READY,
		STATUS_SUSPENDED,
		STATUS_PAUSED,
		STATUS_PLAYING,
		STATUS_ERROR
	};

	//// Constructor
	SoundManager();

	/// Destructor
	~SoundManager();

	/**
	 * Opens an audio device.
	 * @param[in, out] parameters to try when opening device. The hardware may not allow certain paremeters,
	 * in which case the actual parameters used to open the device are used.
	 * @param[in] isSync if true, the device is opened for synchronous play; if false, it is opened for aynschronous play
	 */
	void open(Parameters &params, bool isSync);

	/**
	 * Closes the audio device.
	 */
	void close();

	/**
	 * Returns a read-only pointer to the current audio parameters if device is open, null if it is not
	 * @return audio parameters of the playback device.
	 */
	const Parameters* getParameters() const;

	/**
	 * Speaks out a piece of text using text to speech.
	 * @param aText text to speak out.
	 */
	void speak(const std::string &aText) const;

	/**
	 * Queues a piece of audio to play
	 * The audio is assumed to contain one period's worth of frames, as given by getParameters();.
	 * @param audio pointer to audio data to play. One buffer's worth of data will be read from this location.
	 */
	void play(const TAudioData* audio) const;

	/**
	 * Clears the queue and plays a piece of audio immediately. Any sound currently playing is stopped.
	 * The audio is assumed to contain one buffer's worth of samples, as given by getParameters();.
	 * @param audio pointer to audio data to play. One buffer's worth of data will be read from this location.
	 */
	void playNow(const TAudioData* audio) const;

	/**
	 * Pauses playback without clearing the queue
	 */
	void pause() const;

	/**
	 * Resumes a paused playback
	 */
	void resume() const;

	/**
	 * Stops playback, finishing the playback queue first.
	 */
	void stop() const;

	/**
	 * Stops playback immediately, clearing the playback queue.
	 */
	void stopNow() const;

	/**
	 * Returns the current status of the playback device
	 * @return current sattus of the device
	 */
	Status getStatus() const;

protected:
	std::unique_ptr<AlsaAccess> soundMngr_;
};


#endif /* SOUNDMANAGER_H_ */
