/*
 * BarcodeEngine.cpp
 *
 *  Created on: Sep 14, 2012
 *      Author: kamyon
 */
#include "BarcodeEngine.h"
#include "ski/BLaDE/BLaDE.h"
#include "opts.h"
#include "ski/log.h"
#include "misc.h"
#include <stdexcept>
#include "ski/math.h"

const char* BarcodeEngine::BARCODE_DECODED_TEXT = "Barcode decoded: ";
const char* BarcodeEngine::LOOKUP_TEXT = "Looking up product information";
const char* BarcodeEngine::NO_PRODUCT_FOUND_TEXT = "No product information found";

BarcodeEngine::BarcodeEngine(cv::Mat& input, const Opts &opts)
try:
	input_(input),
	grayImage_(input.size()),
	blade_(new BLaDE(grayImage_, BLaDE::Options(opts.scale))),
	isVisualFeedbackOn_(opts.isWindowShown),
	isAudioFeedbackOn_(opts.isAudioEnabled),
	audioFeedback_(isAudioFeedbackOn_ ? new AudioFeedback() : NULL),
	isProductSearchOn_(opts.isProductLookedUp),
	pSearch_(ProductSearch::create(ProductSearch::Google_Product_Search))
{
	//Add symbologies
	blade_->addSymbology(BLaDE::UPCA);
}
catch (cv::Exception &aErr)
{
	LOGE("Barcode Engine Error! %s\n", aErr.what());
	throw std::runtime_error("Cannot initialize barcode engine");
}
catch (std::exception &aErr)
{
	LOGE("Barcode Engine Error! %s\n", aErr.what());
	throw std::runtime_error("Cannot initialize barcode engine");
}

bool BarcodeEngine::process()
{
	//To use BLaDE w/o opencv, we can write a header over existing data
	//Convert input to grayscale
	cv::cvtColor(input_, grayImage_, CV_BGR2GRAY);

	///-----------------------------------
	///Barcode Location
	///-----------------------------------
	BarcodeList& barcodes = blade_->locate();
	//If no barcode found, done processing this frame and provide nullFeedback, otherwise provide feedback
	if (barcodes.empty())
	{
		if (isAudioFeedbackOn_)
			audioFeedback_->playNull();
		LOGD("No barcodes detected\n");
		return false;	//no barcodes detected
	}

	for (BarcodeList::const_iterator iBC = barcodes.begin(); iBC != barcodes.end(); iBC++)
	{
		if (isVisualFeedbackOn_)
			Draw::line(input_, iBC->firstEdge, iBC->lastEdge, Draw::colorRed, 1);
		LOGD("Barcode found between (%d,%d) and (%d,%d).\n",
				iBC->firstEdge.x, iBC->firstEdge.y, iBC->lastEdge.x, iBC->lastEdge.y);
	}

	///-----------------------------------
	///Barcode Decoding
	///-----------------------------------
	Barcode &bc = barcodes.front(); //barcodes must be sorted, bc is the most dominant barcode seen
	bool isDecoded = blade_->decode(bc);

	///-----------------------------------
	///Feedback
	///-----------------------------------
	if (isDecoded)
	{
		if (isVisualFeedbackOn_)
			Draw::line(input_, bc.firstEdge, bc.lastEdge, Draw::colorGreen, 2);
		LOGD("Decoded barcode between (%d,%d) and (%d,%d) as %s: %s\n",
				bc.firstEdge.x, bc.firstEdge.y, bc.lastEdge.x, bc.lastEdge.y, bc.symbology.c_str(), bc.estimate.c_str());
		outputText(std::string(BARCODE_DECODED_TEXT) + bc.estimate + "\n");
		if (isProductSearchOn_)
		{
			outputText(std::string(LOOKUP_TEXT) + "\n");
			ProductSearch::ProductList products = getProductInfo(bc);
			if (products.empty())
				outputText(std::string(NO_PRODUCT_FOUND_TEXT) + "\n");
			else
			{
				for (ProductSearch::ProductList::const_iterator pProduct = products.begin(); pProduct != products.end(); pProduct++)
					outputText(pProduct->asString() + "\n");
			}
		}
	}
	else
	{
		//Provide audio feedback
		double sizeScore = calculateSizeScore(bc, input_.size());
		double alignmentScore = calculateAlignmentScore(bc, input_.size());
		audioFeedback_->play(sizeScore, alignmentScore);
	}
	return isDecoded;
}

BarcodeEngine::~BarcodeEngine()
{
	delete pSearch_;
}

void BarcodeEngine::outputText(const std::string &aText)
{
	//Print it
	fprintf(stdout, "%s", aText.c_str());
	//Say it
	if (isAudioFeedbackOn_)
		audioFeedback_->say(aText);
	//Show it on image - REMOVED since it gets too cluttered
	/*
	if (isVisualFeedbackOn_)
	{
		static Point imgCenter = Point(input_.size().width >> 1, input_.size().height >> 1);
		Draw::text(input_, imgCenter, aText);
	}
	*/
}

double BarcodeEngine::calculateSizeScore(const Barcode& bc, const TSizeInt& imSize)
{
	double w = imSize.width, h = imSize.height;
	double dX = bc.lastEdge.x - bc.firstEdge.x, dY = bc.lastEdge.y - bc.firstEdge.y;
	double angle = atan(dY / dX);
	double bcSize = sqrt(dX * dX + dY * dY);
	double maxSize = .8 * std::min(w / cos(angle), h / abs(sin(angle)) ), minSize= 0.5 * maxSize;
	//Now calculate the scores
	double sizeScore = 1.0;
	if ( bcSize > maxSize )
		sizeScore  *= maxSize / bcSize;
	else if (bcSize < minSize )
		sizeScore  *= bcSize/ minSize;
	return sizeScore;
}

double BarcodeEngine::calculateAlignmentScore(const Barcode& bc, 	const TSizeInt& imSize)
{
	double w = imSize.width, h = imSize.height;
	int minDist = std::min(w, h) / 20; //how far the edges should be from the edge of the image
	int leftDist = std::min(bc.firstEdge.x, bc.lastEdge.x);
	int rightDist = w - std::max(bc.firstEdge.x, bc.lastEdge.x);
	int topDist = std::min(bc.firstEdge.y, bc.lastEdge.y);
	int botDist = h - std::max(bc.firstEdge.y, bc.lastEdge.y);
	//Now calculate the scores
	double alignmentScore = 1.0;
	if (leftDist < minDist)
		alignmentScore *= .5 * leftDist / minDist + .5;
	if (rightDist < minDist)
		alignmentScore *= .5 * rightDist / minDist + .5;
	if (topDist < minDist)
		alignmentScore *= .5 * topDist / minDist + .5;
	if (botDist < minDist)
		alignmentScore *= .5 * botDist / minDist + .5;
	return alignmentScore;

}

BarcodeEngine::AudioFeedback::AudioFeedback()
try :
	soundParams_(N_CHANNELS, RATE, PERIOD_SIZE, N_PERIODS),
	soundMan_(new SoundManager()),
	sounds_(N_FEEDBACK_LEVELS, N_FEEDBACK_LEVELS)
{
	//Open sound manager for playback
	soundMan_->open(soundParams_, false);
	//generate the sounds
	generateSounds();
	//Start playback - fill buffer
	for (TUInt n = 0; n < soundParams_.nPeriods/2; n++)
		playNull();
}
catch (std::exception &aErr)
{
	LOGE("Audio cannot be initialized: %s\n", aErr.what());
	throw std::runtime_error("AudioFeedback: Cannot initialize audio");
}

BarcodeEngine::AudioFeedback::~AudioFeedback()
{
	for (int i = 0; i < sounds_.rows; i++)
	{
		for (int j = 0; j < sounds_.cols; j++)
			delete [] sounds_(i,j);
	}
}

void BarcodeEngine::AudioFeedback::generateSounds()
{
	LOGD("AudioFeedback: Generating sounds\n")
	//we want to make sure that samples start and end at 0 - thus, we need to readjust the frequency to make sure an integer number of cycles fit in one period
	double nCyclesPerPeriod = 1.0 * soundParams_.periodSize * BASE_FREQUENCY / soundParams_.samplingRate;
	double frequency = BASE_FREQUENCY * (TUInt) nCyclesPerPeriod  / nCyclesPerPeriod;
	double w = 2 * ski::PI * frequency / soundParams_.samplingRate;
	SoundManager::TAudioData volume;
	TUInt64 sampleEnd, t;
	TAudioArray data;

	//Generate null sound
	nullSound_ = new SoundManager::TAudioData[soundParams_.periodSize];
	//Generate feedback sounds
	SoundManager::TAudioData maxVolume = std::numeric_limits<SoundManager::TAudioData>::max();
	for (TUInt x = 0; x < N_FEEDBACK_LEVELS; x++)
	{
		volume = maxVolume >> (N_FEEDBACK_LEVELS - 1 - x);
		for (TUInt y = 0; y < N_FEEDBACK_LEVELS; y++)
		{
			//Reserve space
			sounds_(x,y) = data = new SoundManager::TAudioData[soundParams_.periodSize];
			//Generate samples
			sampleEnd = (y+1) * soundParams_.periodSize / N_FEEDBACK_LEVELS;
			for (t = 0; t < sampleEnd; t++)
				data[t] = volume * std::cos(w * t);
			for (; t < soundParams_.periodSize; t++)
				data[t] = 0;
		}
	}
}

void BarcodeEngine::AudioFeedback::play(double sizeScore, double alignmentScore)
{
	int size = round(sizeScore * (N_FEEDBACK_LEVELS-1));
	int alignment = round(alignmentScore * (N_FEEDBACK_LEVELS-1));
	LOGD("Playing feedback for size = %d, alignment = %d\n", size, alignment);
	play(sounds_(size, alignment));
}

void BarcodeEngine::AudioFeedback::playNull()
{
	play(nullSound_);
}

void BarcodeEngine::AudioFeedback::play(const TAudioArray &aSound)
{
	soundMan_->play(aSound);
}

void BarcodeEngine::AudioFeedback::say(const std::string &aText)
{
	if (soundMan_.get())
		soundMan_->speak(aText);
}

ProductSearch::ProductList BarcodeEngine::getProductInfo(const Barcode& aBarcode)
{
	return pSearch_->identify(aBarcode.estimate);
}
