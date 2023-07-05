#ifndef __MAIN_H
#define __MAIN_H
/**
 * @file Main class
 * @author Ender Tekin
 */

#include "opts.h"
#include "opencv2/opencv.hpp"

using namespace std;

/**
 * Main class
 */
class IOHandler{
public:
	/**
	 * Constructor
	 */
	IOHandler();

	/**
	 * Destructor
	 */
	virtual ~IOHandler();

	/**
	 * Starts the processing.  Based on the input options, initializes input, starts processing frame(s) and then cleans up afterwards.
	 * @return exit code (0 is successful, an error code otherwise).
	 */
	int start();

	/**
	 * Cleanup at the end
	 */
	void cleanup();
private:

	enum EImageMode {EColor = 1, EGray = 0, ENone = -1};

	/**
	 * Initializes the webcam
	 * @return true if camera is successfully initialized
	 * @param[in] aCam index of camera to be initialized
	 */
	void initializeCamera(int aCam=0);

	/**
	 * Initializes images
	 */
	void initializeImageContainers();

	/**
	 * Releases image containers
	 * @param[in] isInputReleased true if input image is also to be released [default: false].
	 */
	void releaseImageContainers(bool isInputReleased=false);

	/**
	 * Reads input image file
	 * @param[in] aFile input file to read.
	 * @param[in] image format (EColor, EGray or ENone, denoting as is) [Default=ENone].
	 * @throw errFileRead if cannot load file.
	 */
	void loadImage(const string &aFile, EImageMode format=ENone);

	/**
	 * Reads input video file
	 * @return true if input video is successfully opened
	 * @param[in] aFile input file to read.
	 * @throw errFileRead if cannot load file.
	 */
	void loadVideo(const string &aFile);

	/**
	 * Decodes video codec returned by opencv into a string
	 * @param aCodec codec value
	 * @return human-readable string containing codec name
	 */
	std::string getVideoCodec();

	/**
	 * Camera loop, runs until isRunning_ = false
	 */
	void loop();

	/**
	 * Gets new frames
	 * @param[in] image image to populate.
	 * @return true if frames successfully retrieved, false if reached end of video file.
	 */
	bool getNewFrame(cv::Mat &image);

	/**
	 * Processes a single frame (camera frame or image file)
	 */
	void processFrame();

	/**
	 * Saves an image to a file
	 * @param[in] image image to save.
	 * @param[in] filename file to save as.
	 * @param[in] params vector of parameters for image compression (See OpenCV documentation).
	 * @throw EFileWriteError if a problem occurred.
	 */
	void saveImage(const cv::Mat &image, const string &filename, const vector<int>& params=vector<int>() );

	/**
	 * Initializes a video writer.
	 * @param[in] filename file name to save as.
	 * @param[in] isColor true if input is color frames.
	 */
	void startVideoWriter(const string &filename, bool isColor=true);

	/**
	 * Saves a video frame to the currently initialized writer.
	 * @param[in] image to save
	 * @return error code from cvWriteFrame
	 */
	void saveFrame(const cv::Mat &image);

	/**
	 * Handles key presses
	 * @param[in] how long to wait for a key press in ms, if 0, waits until keypress [Default=0].
	 * @return true is the key causes termination of the program loop
	 */
	bool checkForKeys(int aTime=0);

	/**
	 * Closes the named window if it exists.  If no name given, close all windows [Default=NULL].
	 * @param[in] name of window to close
	 */
	void closeWindow(const string &aWindow=string());

	//Size of frames
	cv::Size iSize_;

	/** Image container for input image */
	cv::Mat imgRGB;

	/** Capture device */
	cv::VideoCapture iCapture_;

	/** Video writer */
	cv::VideoWriter iWriter_;

	/** frames/sec of the video */
	int fps_;

	/** true if paused */
	bool isPaused_;

	/**
	 * Raises errors based on exceptions thrown
	 * @param[in] aException exception
	 */
	void handleException(const exception &aException);

};

#endif //__MAIN_H
