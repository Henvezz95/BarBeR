
#ifndef STRINGUTILS_H
#define	STRINGUTILS_H

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

namespace artelab
{
    std::vector<std::string>& split(const std::string &s, char delim, std::vector<std::string> &elems);

    std::vector<std::string> split(const std::string &s, char delim);
    
    bool file_exists(std::string filename);
    
    cv::Mat rotate_image(cv::Mat img, int angle, int size_factor=1);
    
    std::string tostring(double i);
    
}
#endif	/* STRINGUTILS_H */

