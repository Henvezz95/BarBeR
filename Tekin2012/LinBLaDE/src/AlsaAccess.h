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
 * @file AlsaAccess.h
 * @author Ender Tekin
 *  Created on: Jul 25, 2012
 */

#ifndef ALSAACCESS_H_
#define ALSAACCESS_H_

#include "ski/Sound/SoundManager.h"
#include <list>
#include <string>
#include <stdexcept>
#include <memory>
#include <alsa/asoundlib.h>

class AlsaAccess
{
public:
	/** Constructor */
	AlsaAccess();

	/** Destructor */
	virtual ~AlsaAccess();

	/** Alsa exception class used by this class for alsa related errors*/
	class AlsaException: public std::exception
	{
	public:
		/**
		 * Constructor
		 * @param msg msg text for debugging
		 * @param err alsa error code that caused this exception
		 */
		explicit AlsaException(const std::string &msg, int err=0):
			msg_( std::string("AlsaException:").append(msg).append("\n").append( (err < 0 ? snd_strerror(err) : "") ) )
		{};
		/** Destructor */
		virtual ~AlsaException() throw() {};
		virtual const char* what() const throw() {return msg_.c_str(); };
	private:
		std::string msg_;
	};

	///Opens audio manager
	void open(SoundManager::Parameters &params, bool isSync);
	///Closes audio manager
	void close();
	///Returns currently used audio paramters
	inline const SoundManager::Parameters* getParameters() const {return (getStatus() == SoundManager::STATUS_CLOSED ? NULL :  &params_);};
	///Speaks out loud a text
	void speak(const std::string &aText);
	///Queues audio data for playback
	void play(const SoundManager::TAudioData *data);
	///Plays audio data now
	void playNow(const SoundManager::TAudioData *data);
	///Pauses playback
	void pause();
	///Resumes playback
	void resume();
	///Stops playback after emptying queue
	void stop();
	///Stops playback immediately
	void stopNow();
	///Returns current status
	SoundManager::Status getStatus() const;

protected:
	typedef snd_ctl_t* card_handle;
	typedef snd_ctl_card_info_t* card_info;
	typedef snd_pcm_t* pcm_handle;
	typedef snd_pcm_uframes_t TULong;
	typedef snd_pcm_sframes_t TLong;

	class SoundCard;
	/**
	 * A wave device that is part of a sound card
	 */
	struct WaveDevice
	{
		/** Index of this device */
		int index;
		/** Hardware name of this device */
		std::string hwName;
		/** Handle to this device */
		pcm_handle handle;
		/** Hardware Parameters of this device */
		snd_pcm_hw_params_t* hwParams;
		/** Software Parameters of this device */
		snd_pcm_sw_params_t* swParams;
		/**
		 * Constructor
		 * @param parent sound card that this device resides on
		 * @param deviceIndex index of this device on the parent card
		 */
		WaveDevice(const SoundCard &parent, int deviceIndex);
		/** Destructor */
		~WaveDevice();
		/**
		 * Open device
		 * @param[in, out] params parameters to use. On return, contains actual parameters that were set.
		 * @param[in] isSycn if true, device is opened for synchronous playback, if false, it is opened for asynchronous playback
		 */
		void open(SoundManager::Parameters &params, bool isSync);
		/** Close device */
		void close();
		/** Device handle */
		typedef std::unique_ptr<WaveDevice> Handle;
	};

	static const WaveDevice::Handle nullDevice;

	/**
	 * A sound card installed in the device
	 */
	struct SoundCard
	{
		/** Index of this card on device */
		int index;
		/** Name of this card */
		std::string name;
		/** Hardware name of this card */
		std::string hwName;
		/** handle to the card if open */
		card_handle handle;
		/** Information regarding this card */
		card_info info;
		/** List of devices on this card */
		std::list<WaveDevice::Handle> devices;
		/** Constructor */
		SoundCard(int cardIndex);
		/** Destructor */
		~SoundCard();
		/** Enumerates devices on this card */
		void enumerateDevices();
		/** Opens card */
		void open();
		/** Closes card and releases information structures */
		void close();
		/** Device handle */
		typedef std::unique_ptr<SoundCard> Handle;
	};

	/** Audio data structure used internally */
	struct private_audio_data
	{
		/** Pointer to the audio data */
		const SoundManager::TAudioData* data;
		/** Number of channels in the data */
		TUInt nChannels;
		/** Number of frames in the data. The data is assumed to be one period-length */
		TUInt64 size;
	};

	/** List of cards and devices found */
	std::list<SoundCard::Handle> cards_;

	/**
	 * Populates the list of sound cards
	 */
	void enumerateSoundCards();

	//-------------
	//Playback stuff
	//--------------
	/** Playback device */
	WaveDevice* playbackDevice_;

	/** Playback parameters */
	SoundManager::Parameters params_;

	/** Playback mode */
	bool isPlaybackSync_;

	/**
	 * Writes data to audio buffer
	 * @param[in] device pcm handle of device to write to
	 * @param[in] data data to write
	 */
	static void write(pcm_handle device, const private_audio_data &data);

	enum Errors
	{
		ERROR_RECOVERABLE,
		ERROR_UNRECOVERABLE,
		ERROR_RETRY,
		ERROR_UNHANDLED
	};
	/**
	 * Recovers from buffer under/overruns.
	 * @param[in] aDev handle of device causing the error
	 * @param[in] err error code that occurred
	 * @return int error code if error is unhandled;
	 */
	static Errors recoverFromError(pcm_handle handle, int error);

	//Asynchronous playback stuff
	/** Asyncronous callback handler structure - we do not access this directly*/
	snd_async_handler_t* asyncHandler_;

	/** Private audio data to be used for playback*/
	private_audio_data privateData_;

	/**
	 * Asynchronous sound callback
	 * @param[in] aHandler sound callback handler
	 */
	static void asyncHandleCallback(snd_async_handler_t *aHandler);

};


#endif /* ALSAACCESS_H_ */
