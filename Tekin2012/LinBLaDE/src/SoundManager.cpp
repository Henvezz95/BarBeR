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
 * @file SoundManager.cpp
 * @author Ender Tekin
 *  Created on: Oct 8, 2012
 */

#include "ski/Sound/SoundManager.h"
#include "AlsaAccess.h"

SoundManager::Parameters::Parameters(TUInt nC/*=2*/, TUInt aRate/*=16000*/, TUInt64 aPerSz/*=4096*/, TUInt nP/*=4*/):
	nChannels(nC),
	samplingRate(aRate),
	periodSize(aPerSz),
	nPeriods(nP),
	bufferSize(0)
{};

TUInt SoundManager::Parameters::frameSizeInBytes() const {return nChannels * sizeof(SoundManager::TAudioData); };

TUInt64 SoundManager::Parameters::periodSizeInBytes() const { return periodSize * frameSizeInBytes(); };

SoundManager::SoundManager(): soundMngr_(new AlsaAccess()) {};

SoundManager::~SoundManager() {};

void SoundManager::open(SoundManager::Parameters &params, bool isSync) {soundMngr_->open(params, isSync); };

void SoundManager::close() {soundMngr_->close(); };

const SoundManager::Parameters* SoundManager::getParameters() const {return soundMngr_->getParameters(); };

void SoundManager::speak(const std::string &aText) const {soundMngr_->speak(aText); };

void SoundManager::play(const SoundManager::TAudioData* audio) const {soundMngr_->play(audio); };

void SoundManager::playNow(const SoundManager::TAudioData* audio) const {soundMngr_->playNow(audio); };

void SoundManager::pause() const {soundMngr_->pause(); };

void SoundManager::resume() const {soundMngr_->resume(); };

void SoundManager::stop() const {soundMngr_->stop(); };

void SoundManager::stopNow() const {soundMngr_->stopNow(); };

SoundManager::Status SoundManager::getStatus() const {return soundMngr_->getStatus(); };
