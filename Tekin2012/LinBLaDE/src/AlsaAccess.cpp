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
 * @file AlsaAccess.cpp
 * @author Ender Tekin
 *  Created on: Jul 30, 2012
 */

#include "ski/log.h"
#include "AlsaAccess.h"
#include "boost/lexical_cast.hpp"

//================================
// Alsa Access main class methods
//================================
const AlsaAccess::WaveDevice::Handle AlsaAccess::nullDevice = AlsaAccess::WaveDevice::Handle();

AlsaAccess::AlsaAccess():
	playbackDevice_(NULL)
{
	//Get device list and find a playback device
	enumerateSoundCards();
	if (playbackDevice_ == NULL)
	{
		LOGE("Alsa could not determine playback device\n");
		throw AlsaException("Could not determine a sound playback device");
	}
	else
		LOGD("Playback device determined as %s\n", playbackDevice_->hwName.c_str());
}

AlsaAccess::~AlsaAccess()
{
	//Stop operations
	stopNow();
}

void AlsaAccess::open(SoundManager::Parameters &params, bool isSync)
{
	playbackDevice_->open(params, isSync);
	params_ = params;
	isPlaybackSync_ = isSync;
	//Set private data - this will not change during playback
	privateData_.nChannels = params_.nChannels;
	privateData_.size = params_.periodSize;
	//set callback function if not set
	if (!isSync)
	{
		int err = snd_async_add_pcm_handler(&asyncHandler_, playbackDevice_->handle, asyncHandleCallback, &privateData_);
		if (err < 0)
			throw AlsaException("Cannot add asynchronous playback handler", err);
	}
	LOGD("Alsa: Opened playback device with parameters:\n#c = %u, fS = %u, per = %lu, #p = %u, bufSz = %lu\n",
			params_.nChannels, params_.samplingRate, params_.periodSize, params_.nPeriods, params_.bufferSize);
}

void AlsaAccess::close()
{
	if (!isPlaybackSync_)
		snd_async_del_handler(asyncHandler_);
	playbackDevice_->close();
	LOGD("Alsa: Closing playback device")
}

void AlsaAccess::speak(const std::string& aText)
{
	//TODO: check is any other audio should be stopped first
	std::string speakCmd("espeak \"" + aText +"\"");
	int err = std::system(speakCmd.c_str());
	if (err != EXIT_SUCCESS)
		throw std::runtime_error("Espeak failed - please check to make sure espeak is installed and properly configured.");
}

void AlsaAccess::play(const TInt16* data)
{
	if (playbackDevice_->handle == NULL)
		throw std::logic_error("Alsa Error: Wave device not open.");
	privateData_.data = data;
	//write data immediately for sync playback or until async playback starts (i.e., buffer is filled)
	if (isPlaybackSync_ || snd_pcm_state(playbackDevice_->handle) != SND_PCM_STATE_RUNNING)
		write(playbackDevice_->handle, privateData_);
}

void AlsaAccess::playNow(const TInt16* data)
{
	//Stop currently playing stream immediately
	stopNow();
	//Play this data
	play(data);
}

void AlsaAccess::pause()
{
	snd_pcm_state_t state = snd_pcm_state(playbackDevice_->handle);
	bool canPause = snd_pcm_hw_params_can_pause(playbackDevice_->hwParams);
	if (canPause && (state == SND_PCM_STATE_RUNNING))
	{
		int err = snd_pcm_pause(playbackDevice_->handle, 0);
		if (err < 0)
			throw AlsaException("Cannot pause audio", err);
	}
}

void AlsaAccess::resume()
{
	snd_pcm_state_t state = snd_pcm_state(playbackDevice_->handle);
	if (state == SND_PCM_STATE_PAUSED)
	{
		int err = snd_pcm_pause(playbackDevice_->handle, 1);
		if (err < 0)
			throw AlsaException("Cannot resume audio", err);
		snd_pcm_drain(playbackDevice_->handle);
	}
}

void AlsaAccess::stop()
{
	snd_pcm_state_t state = snd_pcm_state(playbackDevice_->handle);
	if ( (state == SND_PCM_STATE_RUNNING) || (state == SND_PCM_STATE_PAUSED) )
	{
		snd_pcm_drain(playbackDevice_->handle);
		int err = snd_pcm_prepare(playbackDevice_->handle);
		if (err < 0)
			throw AlsaException("Cannot prepare PCM for use", err);
	}
	else
		LOGE("ALSA already stopped or not open");
}

void AlsaAccess::stopNow()
{
	snd_pcm_state_t state = snd_pcm_state(playbackDevice_->handle);
	if ( (state == SND_PCM_STATE_RUNNING) || (state == SND_PCM_STATE_PAUSED) )
	{
		snd_pcm_drop(playbackDevice_->handle);
		snd_pcm_prepare(playbackDevice_->handle);
	}
	else
		LOGE("ALSA already stopped or not open");
}

SoundManager::Status AlsaAccess::getStatus() const
{
	if (playbackDevice_)
	{
		snd_pcm_state_t state = snd_pcm_state(playbackDevice_->handle);
		switch (state)
		{
		case SND_PCM_STATE_OPEN:
		case SND_PCM_STATE_SETUP:
			throw std::logic_error("Device should not be in this state, since opening should prepare the device");
		case SND_PCM_STATE_DRAINING:
			throw std::logic_error("Device should not be in this state, since only playback mode is implemented");
		case SND_PCM_STATE_PREPARED:
			return SoundManager::STATUS_READY;
		case SND_PCM_STATE_RUNNING:
			return SoundManager::STATUS_PLAYING;
		case SND_PCM_STATE_XRUN:
		case SND_PCM_STATE_DISCONNECTED:
			return SoundManager::STATUS_ERROR;
		case SND_PCM_STATE_PAUSED:
			return SoundManager::STATUS_PAUSED;
		case SND_PCM_STATE_SUSPENDED:
			return SoundManager::STATUS_SUSPENDED;
		default:
			throw std::logic_error("Unhandled device state");
		}
	}
	else
		return SoundManager::STATUS_CLOSED;
}

void AlsaAccess::enumerateSoundCards()
{
	//Counts the available cards
	int cardNum = -1;
	while (true)
	{
		//Count cards
		int err = snd_card_next(&cardNum);
		if (err < 0)
			throw AlsaException("Error enumerating sound cards", err);
		if (cardNum < 0)	//end of cards
			break;
		cards_.push_back(SoundCard::Handle(new SoundCard(cardNum)));
	}
	//Now determine a playback device if any cards have been found
	for (std::list<SoundCard::Handle>::iterator pCard = cards_.begin(); pCard != cards_.end(); pCard++)
	{
		if (!(*pCard)->devices.empty())
		{
			playbackDevice_ = (*pCard)->devices.front().get();
			break;
		}
	}
}

void AlsaAccess::write(pcm_handle device, const AlsaAccess::private_audio_data &data)
{
	//Write data to soundcard buffer
	TInt64 dataLength = data.size, writLength = 0; //in bytes
	while (dataLength > 0)
	{
		const TInt16 *buf = data.data + writLength;	//progress data pointer
		writLength = snd_pcm_writei(device, buf, dataLength);	//get number of actual frames written
		if (writLength < 0)
		{
			Errors err = recoverFromError(device, writLength);
			if (err == ERROR_RETRY)
				continue;
			else if (err == ERROR_UNHANDLED)
				throw AlsaException("Playback error", writLength);
		}
		dataLength -= writLength;	//less data to write
		LOGV("Wrote %ld frames, remaining %ld frames\n", writLength, dataLength);
	}
}

void AlsaAccess::asyncHandleCallback(snd_async_handler_t *aHandler)
{
    static pcm_handle device = snd_async_handler_get_pcm(aHandler);
    static private_audio_data *data = (private_audio_data *) snd_async_handler_get_callback_private(aHandler);
    write(device, *data);
}

AlsaAccess::Errors AlsaAccess::recoverFromError(pcm_handle handle, int error)
{
	Errors err;
    switch(error)
    {
        case -EPIPE:    // Buffer xrun
            LOGE("ALSA ERROR: Buffer xrun\n");
            if ((error = snd_pcm_recover(handle, error, 0)) < 0)
                LOGE("ALSA ERROR: Buffer xrun cannot be recovered: %s\n", snd_strerror(error));
            err = ERROR_UNRECOVERABLE;
            break;

        case -ESTRPIPE: //suspend event occurred
            LOGE("ALSA ERROR: Suspend\n");
            //EAGAIN means that the request cannot be processed immediately
            while ((error = snd_pcm_recover(handle, error, 0)) == -EAGAIN)
                sleep(1);// waits until the suspend flag is clear

            if (error < 0) // error case
            {
                if ((error = snd_pcm_recover(handle, error, 0)) < 0)
                    LOGE("ALSA ERROR: Suspend cannot be recovered: %s\n", snd_strerror(error));
            }
            err = ERROR_UNRECOVERABLE;
            break;

        case -EAGAIN: //Request cannot be processed immediately
        	LOGE("ALSA ERROR: Request cannot be processed immediately\n");
        	err = ERROR_RETRY;
        	break;

        default:
            LOGE("ALSA ERROR: %s\n", snd_strerror(error));
            err =  ERROR_UNHANDLED;
            break;
    }
    return err;
}

//================================
//AlsaAccess::WaveDevice methods
//================================
AlsaAccess::WaveDevice::WaveDevice(const SoundCard &parent, int deviceIndex):
		index(deviceIndex),
		hwName(std::string("plug").append(parent.hwName).append(",").append(1, '0'+(char) index) ),
		handle(NULL),
		hwParams(NULL),
		swParams(NULL)
{

}

AlsaAccess::WaveDevice::~WaveDevice()
{
	close();
}

void AlsaAccess::WaveDevice::open(SoundManager::Parameters &params, bool isSync)
{
	//Get device handle
	if (handle != NULL)
		throw std::logic_error("Alsa: Device already open");
	int err = snd_pcm_open(&handle, hwName.c_str(), SND_PCM_STREAM_PLAYBACK, 0);
	//int err = snd_pcm_open(&handle, hwName.c_str(), SND_PCM_STREAM_PLAYBACK, isSync ? SND_PCM_NONBLOCK : SND_PCM_ASYNC);
	if (err < 0)
		throw AlsaException(std::string("Cannot open device ").append(hwName), err);
	//Device should now be in SND_PCM_STATE_OPEN

	//Set device hardware parameters:
	//Allocate room for the parameters
	err = snd_pcm_hw_params_malloc(&hwParams);
	if (err < 0)
		throw AlsaException("Cannot allocate room for device parameters.", err);
	//Get parameters;
	err = snd_pcm_hw_params_any(handle, hwParams);
	if (err < 0)
		throw AlsaException("Cannot get device parameters.", err);
	//Set the parameters to new ones;
	//Default parameters:
	err = snd_pcm_hw_params_set_format(handle, hwParams, SND_PCM_FORMAT_S16);	//16-bit signed data
	if (err < 0)
		throw AlsaException("Cannot set data format to signed 16-bit", err);
	err = snd_pcm_hw_params_set_access(handle, hwParams, SND_PCM_ACCESS_RW_INTERLEAVED); //interleaved channels
	if (err < 0)
		throw AlsaException("Cannot set access.", err);
	//User parameters:
	err = snd_pcm_hw_params_set_rate_near(handle, hwParams, &params.samplingRate, 0);
	if (err < 0)
		throw AlsaException("Cannot set sampling rate.", err);
	err = snd_pcm_hw_params_set_channels(handle, hwParams, params.nChannels);
	if (err < 0)
		throw AlsaException("Cannot set number of channels.", err);
	err = snd_pcm_hw_params_set_periods(handle, hwParams, params.nPeriods, 0);
	if (err < 0)
		throw AlsaException("Cannot set number of periods.", err);
	err = snd_pcm_hw_params_set_period_size_near(handle, hwParams, &params.periodSize, 0);
	if (err < 0)
		throw AlsaException("Cannot set period size.", err);
	//Get buffer size
	err = snd_pcm_hw_params_get_buffer_size(hwParams, &params.bufferSize);
	if (err < 0)
		throw AlsaException("Cannot get buffer size.", err);
	//Apply parameters to device;
	err = snd_pcm_hw_params(handle, hwParams);
	if (err < 0)
		throw AlsaException("Cannot apply hardware parameters to device.", err);

	//Set device software parameters:
	//Allocate room for the parameters
	err = snd_pcm_sw_params_malloc(&swParams);
	if (err < 0)
		throw AlsaException("Cannot allocate room for device parameters.", err);
	err = snd_pcm_sw_params_current(handle, swParams);
	if (err < 0)
		throw AlsaException("Cannot get device parameters.", err);
	//Set the parameters to new ones;
	err = snd_pcm_sw_params_set_start_threshold(handle, swParams, params.bufferSize - params.periodSize); 	//Start playback when buffer is almost full
	if (err < 0)
		throw AlsaException("Cannot set start threshold", err);
	err = snd_pcm_sw_params_set_avail_min(handle, swParams, isSync ? params.bufferSize : params.periodSize); 	// allow the transfer when at least period_size samples can be processed or disable this mechanism when period event is enabled (aka interrupt like style processing)
	if (err < 0)
		throw AlsaException("Cannot set notification threshold", err);
	if (!isSync)
	{
		err = snd_pcm_sw_params_set_period_event(handle, swParams, 1); 	// enable period events when requested
		if (err < 0)
			throw AlsaException("Unable to set period event.", err);
	}
	// write the parameters to the playback device
	err = snd_pcm_sw_params(handle, swParams);
	if (err < 0)
		throw AlsaException("Cannot apply software parameters to device.", err);

	//Check that the parameters have been successfully applied
	snd_pcm_state_t state = snd_pcm_state(handle);
	if (state != SND_PCM_STATE_PREPARED)
		throw AlsaException("Device parameters cannot be applied", 0);

}

void AlsaAccess::WaveDevice::close()
{
	if (handle != NULL)
	{
		//Stop playback
		snd_pcm_drop(handle);
		snd_pcm_prepare(handle);
		int err = snd_pcm_close(handle);
		if (err < 0)
			throw AlsaException("Cannot release wave device " + hwName, err);
		handle = NULL;
		if (hwParams != NULL)
		{
			snd_pcm_hw_params_free(hwParams);
			hwParams = NULL;
		}
		if (swParams != NULL)
		{
			snd_pcm_sw_params_free(swParams);
			swParams = NULL;
		}
	}
}

//================================
//AlsaAccess::SoundCard methods
//================================
AlsaAccess::SoundCard::SoundCard(int cardIndex):
		index(cardIndex),
		hwName("hw:" + boost::lexical_cast<std::string>(cardIndex)),
		handle(NULL),
		info(NULL)
{
	//Open card
	open();
	//enumerate card devices
	enumerateDevices();
	//get device information
	// Tell ALSA to fill in info about this card
	int err = snd_ctl_card_info_malloc(&info);
	if (err < 0)
		throw AlsaException("Cannot allocate card information structure.", err);
	err = snd_ctl_card_info(handle, info);
	if (err < 0)
		throw AlsaException("cannot get card info.", err);
	//close card
	close();
}

AlsaAccess::SoundCard::~SoundCard()
{
	//Release card if still open
	try
	{
		close();
	}
	catch (AlsaException &err) //so that destructor does not throw exception
	{
		LOGE("%s\n", err.what());
	}
	//Release sound card info
	if (info != NULL)
	{
		snd_ctl_card_info_free(info);
		info = NULL;
	}
}

void AlsaAccess::SoundCard::enumerateDevices()
{
	if (handle == NULL)
		throw std::logic_error("Alsa: Card must be open before devices can be detected.");
	devices.clear();
	//Find the wave devices for this card
	int devNum = -1;
	while (true)
	{
		//Count devices for this card
		int err = snd_ctl_pcm_next_device(handle, &devNum);
		if (err < 0)
			throw AlsaException("Error enumerating devices", err);
		if (devNum < 0) //end of devices
			break;
		devices.push_back(WaveDevice::Handle(new WaveDevice(*this, devNum)));
	}
}

void AlsaAccess::SoundCard::open()
{
	//Open card
	if (handle != NULL)
		throw std::logic_error("Alsa: Sound card is already opened.");
	int err = snd_ctl_open(&handle, hwName.c_str(), 0);
	if (err < 0)
		throw AlsaException("Cannot open sound card.", err);
}

void AlsaAccess::SoundCard::close()
{
	if (handle != NULL)
	{
		//Close sound card
		int err = snd_ctl_close(handle);
		if (err < 0)
			throw AlsaException("Cannot release sound card", err);
		handle = NULL;
	}
}
