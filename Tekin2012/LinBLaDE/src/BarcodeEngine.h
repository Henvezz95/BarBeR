/*
 * BarcodeEngine.h
 *
 *  Created on: Sep 14, 2012
 *      Author: kamyon
 */

#ifndef BARCODEENGINE_H_
#define BARCODEENGINE_H_

#include "ski/cv.hpp"
#include "ski/math.h"
#include "ski/BLaDE/Barcode.h"
#include "ski/Sound/SoundManager.h"
#include <cstdio>
#include "ProductSearch.h"
#include "opencv2/opencv.hpp"
#include <memory>

class Opts;
class BLaDE;

/**
 * @class Barcode Feedback and BLaDE access wrapper
 */
class BarcodeEngine
{
public:
	/**
	 * Constructor
	 * @param input image matrix to process
	 * @param opts options to use
	 */
	BarcodeEngine(cv::Mat& input, const Opts &opts);
	/**
	 * Processes current frame
	 * @return true if a barcode is found.
	 */
	bool process();
	/**
	 * Destructor
	 */
	~BarcodeEngine();
private:
	//=======================
	// BLaDE Access
	//=======================
	/** input matrix reference */
	cv::Mat &input_;
	/** Grayscale version of input image */
	TMatrixUInt8 grayImage_;
	/** Handle of the actual localization and decoding engine */
	std::unique_ptr<BLaDE> blade_;

	//=======================
	// FEEDBACK
	//=======================
	/** Text to output when a barcode is decoded */
	static const char* BARCODE_DECODED_TEXT;
	/** Text to output when looking up product information */
	static const char* LOOKUP_TEXT;
	/** Text to output when no product information is found */
	static const char* NO_PRODUCT_FOUND_TEXT;

	/**
	 * Outputs a text
	 * @param aText
	 */
	void outputText(const std::string &aText);

	/**
	 * Calculates the size score of a barcode, indicating how happy we are with the barcode's size
	 * @param bc detected barcode
	 * @param imSize size of image barcode is detected in
	 * @return a score showing indicating how appropriate the size of the barcode is for attempting decoding
	 */
	static double calculateSizeScore(const Barcode &bc, const TSizeInt& imSize);

	/**
	 * Calculates the alignment score of a barcode, indicating how happy we are with the barcode's
	 * alignment in the image frame.
	 * @param bc detected barcode
	 * @param imSize size of image barcode is detected in
	 * @return a score showing indicating how well aligned the barcode is in the image for attempting decoding
	 */
	static double calculateAlignmentScore(const Barcode &bc, const TSizeInt& imSize);

	//-----------------------
	//Visual Feedback
	//-----------------------
	/** True if visual feedback is on */
	bool isVisualFeedbackOn_;

	//-----------------------
	//Audio Feedback
	//-----------------------
	/** True if audio feedback is on */
	bool isAudioFeedbackOn_;

	class AudioFeedback
	{
	private:
		/**
		 * Parameters to use for audio feedback
		 */
		SoundManager::Parameters soundParams_;

		/** Handle of the sound manager used to provide audio feedback */
		std::unique_ptr<SoundManager> soundMan_;

		/** Number of feedback levels */
		static const TUInt N_FEEDBACK_LEVELS = 5;
		/** Number of audio channels */
		static const TUInt N_CHANNELS = 1;
		/** Sampling rate */
		static const TUInt RATE = 16000;
		/** Audio snippet size */
		static const TUInt64 PERIOD_SIZE = 4096;
		/** Number of snippets in buffer */
		static const TUInt64 N_PERIODS = 4;
		/** Approximate audio frequency. Actual frequency may differ slightly due to buffering considerations */
		static const TUInt BASE_FREQUENCY = 800;
		/// Audio array
		typedef SoundManager::TAudioData* TAudioArray;

		/** Pre-generated audio samples to play */
		cv::Mat_<TAudioArray> sounds_;

		/** Pre-generated quiet sound (NULL feedback) */
		TAudioArray nullSound_;
	public:
		/// Constructor
		AudioFeedback();

		/// Destructor
		~AudioFeedback();

		/**
		 * Plays an audio sound if sound feedback is turned on
		 * @param aSound sound data to play
		 */
		void play(double sizeScore, double alignmentScore);

		/**
		 * Plays null sound
		 */
		void playNull();

		/**
		 * Uses text-to-speech to present a text
		 */
		void say(const std::string &aText);
	private:

		/**
		 * Plays a given sound data
		 * @param aSound sound data to play
		 */
		void play(const TAudioArray &aSound);

		/**
		 * Generates the sounds
		 */
		void generateSounds();
	};

	std::unique_ptr<AudioFeedback> audioFeedback_;

	//=======================
	// PRODUCT LOOKUP
	//=======================
	/** Whether product information is looked up online after decoding */
	bool isProductSearchOn_;

	/**
	 * Looks up product information based on decoded barcode
	 * @return product information
	 */
	ProductSearch::ProductList getProductInfo(const Barcode &aBarcode);

	ProductSearch* pSearch_;

};


#endif /* BARCODEENGINE_H_ */
