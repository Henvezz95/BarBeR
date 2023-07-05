/**
 * Main handler class
 */

#include "iohandler.h"
#include "opts.h"
//#include "LinKeys.h"
#include "boost/lexical_cast.hpp"
#include <stdexcept>
#include "misc.h"
#include "BarcodeEngine.h"

extern Opts gOpts;

IOHandler::IOHandler() :
	iSize_(320 * gOpts.resolution, 240 * gOpts.resolution),
	iCapture_(),
	iWriter_(),
	fps_(0),
	isPaused_(false)
{
}

IOHandler::~IOHandler()
{
	cleanup();
}

int IOHandler::start()
{
	///Main body of function.
	try
	{
		///First initialize inputs
		switch (gOpts.input)
		{
		case Opts::EInputWebcam:
			initializeCamera(gOpts.camera);
			break;
		case Opts::EInputMovie:
			loadVideo(gOpts.inputFile);
			break;
		case Opts::EInputImage:
			loadImage(gOpts.inputFile);
			break;
		}
		///Initialize image containers
		initializeImageContainers();
		///Start the main loop.  For videos, new frames will continuously be read.
		loop();
	}
	catch (const exception &aErr)
	{
		handleException(aErr);
		return EXIT_FAILURE;
	}
	///Return error code if any errors were encountered.
	return EXIT_SUCCESS;
}

void IOHandler::loadImage(const string &aFile, EImageMode format)
{
	imgRGB = cv::imread(aFile, (int) format);
	if (imgRGB.data == NULL)
		throw runtime_error("Error loading image file " + aFile);
	iSize_ = imgRGB.size();
	LOGV("Input: (File %s) is %dx%d image\n", aFile.c_str(), iSize_.width, iSize_.height);
}

void IOHandler::loadVideo(const string &aFile)
{
	//read input file;
	if (!iCapture_.open(aFile.c_str()))
		throw runtime_error("Error loading video file " + aFile);
	//Determine video info
	//Get video resolution
	iSize_.height = (int) iCapture_.get(CV_CAP_PROP_FRAME_HEIGHT);
	iSize_.width = (int) iCapture_.get(CV_CAP_PROP_FRAME_WIDTH);
	//Get codec
	string codecName = getVideoCodec();
	LOGV("Input: (File: %s) is %4dx%4d %s coded video @ %2d fps.\n", aFile.c_str(), iSize_.width, iSize_.height, codecName.c_str(),  (int) iCapture_.get(CV_CAP_PROP_FPS));
}

std::string IOHandler::getVideoCodec()
{
	int codecD = (int) iCapture_.get(CV_CAP_PROP_FOURCC);
	/*
	char codec[5];
	codec[0] = codecD & 255;
	codec[1] = (codecD >> 8) & 255;
	codec[2] = (codecD >> 16) & 255;
	codec[3] = (codecD >> 24) & 255;
	codec[4] = '\0';
	*/
	char* codecN = (char*) &codecD;
	string codecName(codecN, codecN+4);
	return codecName;
}

void IOHandler::initializeCamera(int aCam/*=0*/)
{
	/// open the default camera
	if (!iCapture_.open(aCam)) // check if we succeeded
		throw runtime_error("Error initializing camera");

	///Set resolution.  Default resolution is 320x240
	LOGD("Setting camera %d resolution to %4dx%4d\n", aCam, iSize_.width, iSize_.height);
	iCapture_.set(CV_CAP_PROP_FRAME_WIDTH, iSize_.width);
	iCapture_.set(CV_CAP_PROP_FRAME_HEIGHT, iSize_.height);
	///Get actual camera resolution
	iSize_.width = (int) iCapture_.get(CV_CAP_PROP_FRAME_WIDTH);
	iSize_.height = (int) iCapture_.get(CV_CAP_PROP_FRAME_HEIGHT);

	LOGD("Input: (Camera %d) resolution: %4dx%4d.\n", aCam, iSize_.width, iSize_.height);
}

void IOHandler::loop()
{
	int keyWaitTime = (gOpts.input == Opts::EInputImage ? 0 : 10); //wait for key press if input is still image
	int frameCt = 0;
	LOGD("Starting loop.\n");
	while (true)
	{
		if (isPaused_ && !checkForKeys())	//pause unless asked to quit
			break;
		frameCt++;
		///--------------
		/// Get new frame if video
		///--------------
		if (gOpts.input != Opts::EInputImage)
		{
			try
			{
				if (!getNewFrame(imgRGB))
					break;
			}
			catch (exception &aErr)
			{
				throw;
			}
		}

		///--------------
		///Process Frame
		///--------------
		try
		{
			processFrame();
			if (gOpts.input == Opts::EInputImage)
				isPaused_ = true;
		}
		catch (exception &aErr)
		{
			LOGE("Exception processing frame %d:\n", frameCt);
			throw;
		}

		///--------------
		///Display Output
		///--------------
		if (gOpts.isWindowShown)
			imshow("Original Image", imgRGB);
		else if (gOpts.input == Opts::EInputImage)
			break;
		///--------------
		///Check for key presses or quit if it is an image and window is not shown
		///--------------
		if (!checkForKeys(keyWaitTime))
			break;
	}
}

bool IOHandler::getNewFrame(cv::Mat &image)
{
	iCapture_ >> image;
	bool frameRetrieved = (image.data != NULL);
	if (!frameRetrieved)
	{
		double pos = iCapture_.get(CV_CAP_PROP_POS_AVI_RATIO);
		LOGD("Input is %f complete.\n", 100 * pos);
		if (pos < 1) //not end of movie yet
			throw runtime_error("Cannot retrieve new frame\n");
	}
	return frameRetrieved;
}

void IOHandler::processFrame()
{
	try
	{
		//Prepare the barcode engine
		static BarcodeEngine barcodeEngine(imgRGB, gOpts);
		isPaused_ = barcodeEngine.process();
	}
	catch (cv::Exception &aErr)
	{
		LOGE("BLaDE Exception: %s", aErr.what());
		throw;
	}
}

void IOHandler::initializeImageContainers()
{
	//Also used when resizing
	releaseImageContainers();
}

void IOHandler::releaseImageContainers(bool isInputReleased/*=false*/)
{
	if (isInputReleased && imgRGB.data) //if previously initialized
		imgRGB.release();
}

void IOHandler::cleanup()
{
	///Release image containers
	//releaseImageContainers(true);
	///Destroy all windows
	closeWindow();
}

void IOHandler::closeWindow(const string &aWindow/*=string()*/)
{
	if (aWindow.empty())
	{
		///Close all windows
		cvDestroyAllWindows();
	}
	else
	{
		///Check to make sure window exists
		//HWND hWin = (HWND) cvGetWindowHandle(aWindow.c_str()); //for Windows
		void *hWin = cvGetWindowHandle(aWindow.c_str());
		if (hWin != NULL) //window exists
			cvDestroyWindow(aWindow.c_str());
	}
}

void IOHandler::saveImage(const cv::Mat &image, const string &filename,
		const vector<int>& params/*=vector<int>()*/)
{
	if (!imwrite(filename, image, params))
		throw runtime_error("Error saving image " + filename);
}

void IOHandler::startVideoWriter(const string &filename, bool isColor)
{
	int codec = 0; //CV_FOURCC('D','I','V','X');
	cv::Size aSize(iSize_.height, iSize_.width);
	if (!iWriter_.open(filename, codec, fps_, aSize, isColor))
		throw runtime_error("Error initializing video writer");
}

void IOHandler::saveFrame(const cv::Mat &image)
{
	if (iWriter_.isOpened())
		iWriter_ << image;
	else
		throw logic_error("Video writer not open");
}

bool IOHandler::checkForKeys(int aTime)
{
	static TUInt ct = 0;
	int key = cvWaitKey(aTime) & 255;
	switch (key)
	{
	case 255: //no key pressed
		break;
	case 0x1B:	//Escape
	case 0x0A:	//Enter
		LOGV("\n");
		return false;
		break;
	case 'h':
	case 'H':
	case '?':
		LOGV("Help text here...\n");
		break;
	case 'p':
	case 'P':
		isPaused_ = !isPaused_;
		break;
	case 's':
	case 'S':
		saveImage(imgRGB, "BLaDe_" + boost::lexical_cast<string>(ct++) + ".png");
		break;
	default:
		LOGD("Key pressed = %x\n", key);
		break;
	}
	return true;
}

void IOHandler::handleException(const exception &aException)
{
	LOGE("ERROR: %s\n", aException.what());
}
