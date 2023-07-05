#ifndef __OPTS_H
#define __OPTS_H
/**
 * @file options
 */

#include "ski/types.h"
#include <string>
#include <cstdio>

struct Opts
{
	/** input mode */
	enum InputMode
	{
		EInputImage, EInputMovie, EInputWebcam
	} input;
	/** Resolution */
	enum CameraResolution
	{
		EResolutionLow = 1, //320x240
		EResolutionMed, //640x480
		EResolutionHi  //960x720
	} resolution;
	/** Whether windows should be shown or not */
	bool isWindowShown;
	/** Product lookup info on/off */
	bool isProductLookedUp;
	/** Whether audio is enabled */
	bool isAudioEnabled;
	/** Filename of input (empty if input is webcam).*/
	std::string inputFile;
	/** index of webcam if input is webcam (0: any, 1: cam 1, 2: cam 2). */
	int camera;
	/** Scale used for the finder */
	TUInt scale;
	/** Constructor - also sets default values */
	Opts() :
		input(EInputWebcam),
		resolution(EResolutionMed),
		isWindowShown(true),
		isProductLookedUp(false),	//TODO: after implementing the lookup, change this to true
		isAudioEnabled(true),
		camera(0),
		scale(0)
	{
	};
};

/**
 * Parses the command line arguments and sets the option structure given in the parameter
 * @param[in] argc number of arguments.
 * @param[in] argv list of agument strings passed.  argv[0] is the name of the program itself.
 * @param[in] opts structure that on return contains the options updated from the command line
 * @return true if program should continue, false if program should quit gracefully
 * @throw errParser if parser has encountered an error.
 */
bool parse(int argc, char* argv[], Opts &opts);

#endif //__OPTS_H
