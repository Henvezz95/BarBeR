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
/**
 * Main entry point
 * @author Ender Tekin
 */

#include <BLaDE_jni.h>
#include "ski/BLaDE/BLaDE.h"

enum ReturnCodes
{
	RES_JNIERR = ski_blade_FrameProcessor_Status_ERROR,
	RES_BLADEERR = ski_blade_FrameProcessor_Status_ERROR,
	RES_BCDETECTED = ski_blade_FrameProcessor_Status_DETECTED,
	RES_BCDECODED = ski_blade_FrameProcessor_Status_DECODED,
	RES_BCDECODINGFAILED = ski_blade_FrameProcessor_Status_DECODING_FAILED,	///currently not being used
	RES_NULL = ski_blade_FrameProcessor_Status_NOT_FOUND
};

jint Java_ski_blade_FrameProcessor_00024NativeProcessor_blade(JNIEnv* env, jobject obj,
		jbyteArray yuv420, jint height, jint width, jobject barcode, jboolean locationOnly)
{
	int scale = 0;
	//========================
	//Get input
	//========================
	if ((height == 480) && (width == 640))
		scale = 1;
	else if ((height == 240) && (width == 320))
		scale = 0;
	else
		return RES_JNIERR;
	jbyte* in = env->GetByteArrayElements(yuv420, NULL);
	if (in == NULL)
		return RES_JNIERR;
	//Convert Y data of YUV input to grayscale matrix
	static TMatrixUInt8 inputImg(height, width);
	TUInt8 *data = inputImg.data, *input = (TUInt8*) in;
	TUInt sz = height * width;
	TUInt8 *inputEnd = input + sz;
	for (; input < inputEnd; input++, data++)
		*data = (*input > 16 ? *input - 16 : 0);
	int res;
	try
	{
		static bool isInitialized = false;
		///Create engine
		static BLaDE blade(inputImg, BLaDE::Options(scale));
		if (!isInitialized)
		{
			blade.addSymbology(BLaDE::UPCA);
			LOGD("Native blade initialized at scale %d\n", scale);
			isInitialized = true;
		}
		///Attempt to locate barcode
		BarcodeList& barcodes = blade.locate();
		//If no barcode found, done processing this frame
		if (barcodes.empty())
			res = RES_NULL;
		else
		{
			//Barcode detected, return location
			Barcode &bc = barcodes.front(); //barcodes are sorted, bc is the most dominant barcode seen
			//Return the barcode information transformed landscape -> portrait
			TPointInt pt1 = bc.firstEdge, pt2 = bc.lastEdge;
			jclass cls = env->GetObjectClass(barcode);
			jfieldID fID = env->GetFieldID(cls, "x1", "I");
			env->SetIntField(barcode, fID, pt1.x);
			fID = env->GetFieldID(cls, "y1", "I");
			env->SetIntField(barcode, fID, pt1.y);
			fID = env->GetFieldID(cls, "x2", "I");
			env->SetIntField(barcode, fID, pt2.x);
			fID = env->GetFieldID(cls, "y2", "I");
			env->SetIntField(barcode, fID, pt2.y);
			LOGD("Found barcode between (%d,%d)-(%d,%d)", pt1.x, pt1.y, pt2.x, pt2.y);
			res = RES_BCDETECTED;
			///Attempt to decode barcode
			if (!locationOnly)
			{
				bool isDecoded = blade.decode(bc);
				if (isDecoded)
				{
					//Return the barcode information
					fID = env->GetFieldID(cls, "UPC", "Ljava/lang/String;");
					jstring upc = env->NewStringUTF(bc.estimate.c_str());
					env->SetObjectField(barcode, fID, upc);
					res = RES_BCDECODED;
				}
			}
		}

	}
	catch (std::exception &err)
	{
		LOGE("Native Library Error: %s\n", err.what());
		res = RES_BLADEERR;
	}
	env->ReleaseByteArrayElements(yuv420, in, NULL);
	LOGD("Returning with code %d\n", res);
	return res;
}
