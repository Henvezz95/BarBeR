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
 *  log.h
 *  Debugging macros
 *  Created on: Jul 30, 2012
 *      Author: kamyon
 */

#ifndef ELOG_H_
#define ELOG_H_

#ifdef __ANDROID__	//if building for android

#include <android/log.h>

const char APP_NAME[] = "SKI APP";

#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, APP_NAME, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG  , APP_NAME, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO   , APP_NAME, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN   , APP_NAME, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR  , APP_NAME, __VA_ARGS__)

#else // __ANDROID__ not defined -> building for a pc system

#include <cstdio>
#include <cstdarg>

#ifdef DEBUG
#define LOGD(...) {std::fprintf(stderr, __VA_ARGS__);}
#define LOGV(...) {std::fprintf(stdout, __VA_ARGS__);}
#else
#define LOGD(...) {;}
#define LOGV(...) {;}
#endif	//DEBUG_

#define LOGE(...) {std::fprintf(stderr, "ERROR: "); std::fprintf(stderr, __VA_ARGS__);}
#define LOGW(...) {std::fprintf(stderr, "WARNING: "); std::fprintf(stderr, __VA_ARGS__);}
#define LOG(...) {std::fprintf(stdout, __VA_ARGS__);}

//Workaround for curses
#ifdef CURSES
extern "C"
{
#undef LOG
#define LOG(...) {printw(__VA_ARGS__);}
#undef LOGD
#define LOGD(...) {printw(__VA_ARGS__);}
#undef LOGV
#define LOGV(...) {printw(__VA_ARGS__);}
#undef LOGE
#define LOGE(...) {printw(__VA_ARGS__);}
}
#endif //CURSES

#endif //__ANDROID__
#endif //ELOG_H_
