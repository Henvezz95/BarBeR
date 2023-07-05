/**
 * @file File that parses the command line parameters and updates the default options in opts.h
 */

#include "opts.h"
#include "boost/program_options.hpp"
#include <string>
#include <iostream>
#include <cstdio>

namespace po = boost::program_options;
using namespace std;

bool parse(int ac, char* av[], Opts &opts)
{
    //From http://www.boost.org/doc/libs/1_40_0/doc/html/program_options/tutorial.html
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("version,v", "prints version info")
        ("find,f", "looks up product information online")
        ("log,l", "logs the messages to .log and .dbg files instead of stderr and stdout")
        ("quiet,q","turn off audio feedback")
        ("no-show","does not show the input")
        ("debug,d", po::value<int>(), "set debug level")
        ("resolution,r", po::value<int>(), "set resolution level")
        ("input,i", po::value<string>(), "input file or camera index")
        ("scale,s", po::value<int>(), "set working scale")
        ("threshold,t", po::value<int>(), "set gradient threshold")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);
    if (vm.count("help"))
    {
        cout << desc << endl;
        return false;
    }
    if (vm.count("version"))
    {
        cout << "Version: 0.9a, built " << __DATE__ << " " << __TIME__ << endl;
#ifdef COMMIT_VERSION
        cout << "Revision: r" << COMMIT_VERSION << ", committed: " << COMMIT_DATE << endl;
#endif
        return false;
    }
    if (vm.count("find"))
    {
        opts.isProductLookedUp = true;
    }
    if (vm.count("quiet"))
    {
    	opts.isAudioEnabled = false;
    }
    if (vm.count("no-show"))
    {
    	opts.isWindowShown = false;
    }
    if (vm.count("log"))
    {
        std::freopen("BLaDE.log","w",stderr);
        std::freopen("BLaDE.out","w",stdout);
    }
    if (vm.count("scale"))
    {
        int s = vm["scale"].as<int>();
        if ((s<0) || (s > 3))
            throw domain_error("--scale out of range [0..3]");
        opts.scale = s;
    }
    if (vm.count("resolution"))
    {
        int r = vm["resolution"].as<int>();
        switch (r)
        {
        case 0:
        	opts.resolution = Opts::EResolutionLow;
        	break;
        case 1:
        	opts.resolution = Opts::EResolutionMed;
        	break;
        case 2:
        	opts.resolution = Opts::EResolutionHi;
        	break;
        default:
            throw domain_error("--resolution out of range [0..3]");
        }
    }
    if (vm.count("input"))
    {
        string s = vm["input"].as<string>();
        if (s.compare("0") == 0)
        {
            opts.camera = 0;
            opts.input = Opts::EInputWebcam;
        }
        else if (s.compare("1") == 0)
        {
            opts.camera = 1;
            opts.input = Opts::EInputWebcam;
        }
        else if (s.compare("2") == 0)
        {
            opts.camera = 2;
            opts.input = Opts::EInputWebcam;
        }
        else
        {
            if (s.length() < 4)
                throw invalid_argument("--input filename should have a 3 character extension.");
            string sExt = s.substr(s.length()-4);
            if ( ( sExt.compare(".avi") && sExt.compare(".mpg") && sExt.compare(".mp4")) == 0)
            {
                opts.inputFile = s;
                opts.input = Opts::EInputMovie;
            }
            else if ( ( sExt.compare(".jpg") && sExt.compare(".bmp") && sExt.compare(".png")) == 0)
            {
                opts.inputFile = s;
                opts.input = Opts::EInputImage;
                opts.isAudioEnabled = false;
            }
            else
                throw invalid_argument("--input should either be 0,1,2 for the webcam(s), or have one of .mpg, .avi, .mp4, .jpg, .bmp, .png extensions.");
        }
    }
    return true;
};
