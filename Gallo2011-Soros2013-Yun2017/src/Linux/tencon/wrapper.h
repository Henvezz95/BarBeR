#ifndef BARCODE_WRAPPER
#define BARCODE_WRAPPER

#if defined _WIN32 || defined __CYGWIN__
  #ifdef BUILDING_DLL
    #ifdef __GNUC__
      #define DLL_PUBLIC __attribute__ ((dllexport))
    #else
      #define DLL_PUBLIC __declspec(dllexport) 
    #endif
  #else
    #ifdef __GNUC__
      #define DLL_PUBLIC __attribute__ ((dllimport))
    #else
      #define DLL_PUBLIC __declspec(dllimport) 
    #endif
  #endif
  #define DLL_LOCAL
#else
  #if __GNUC__ >= 4
    #define DLL_PUBLIC __attribute__ ((visibility ("default")))
    #define DLL_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define DLL_PUBLIC
    #define DLL_LOCAL
  #endif
#endif

#include <vector>
#include <memory>
#include <string>
#include <stdlib.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include "gallo/gallo.h"
#include "soros/soros.h"
#include "yun/yun.h"

extern "C" DLL_PUBLIC void galloProcess(int* result, unsigned char* img_color, int h, int w, int WinSz);

extern "C" DLL_PUBLIC void sorosProcess(int* result, unsigned char* img_color, int h, int w, int WinSz);

extern "C" DLL_PUBLIC void yunProcess(int* result, int* num_results, unsigned char* img_color, int h, int w);

#endif //BARCODE_WRAPPER