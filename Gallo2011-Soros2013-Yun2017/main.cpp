#include <string>
#include <stdlib.h>
#include <iostream>

#include "gallo/gallo.h"
#include "soros/soros.h"
#include "yun/yun.h"

#include "wrapper.h"

// key
const char* keys = 
    "{h        help |                      | print help message                            }"
    "{file          | /file/dir/file_name  | test image file(.bmp .jpg .png)               }";

int main(int argc, char* argv[])
{
	cv::CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help") || !cmd.check())
    {
		cmd.printMessage();
		cmd.printErrors();
		return 0;
    }
	
	std::string fn = cmd.get<std::string>("file");
	fn = "/media/sf_evezzali/OneDrive - Unimore/PhD Resources/Datasets/Artelab 1D barcodes/Dataset1/05102009101.jpg";
	std::cout<< fn.c_str() <<std::endl;
	
	iy::Gallo mGallo;
	iy::Soros mSoros;
	iy::Yun mYun;

	cv::Mat frame_gray;
	cv::Mat frame = cv::imread(fn.c_str());
	
	if(frame.data == NULL) {
		std::cerr << "error! read image" << std::endl;
		return -1;
	}

	cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

	cv::Rect g_rt = mGallo.process(frame_gray, 20);
	cv::rectangle(frame, g_rt, cv::Scalar(0, 255, 0), 2);

	cv::Rect s_rt = mSoros.process(frame_gray, true, 20);
	cv::rectangle(frame, s_rt, cv::Scalar(255,0,0), 2);

	std::vector<iy::YunCandidate> list_barcode = mYun.process(frame_gray);
	if (!list_barcode.empty())
	{
		for (std::vector<iy::YunCandidate>::iterator it = list_barcode.begin(); it < list_barcode.end(); it++)
		{
			if (it->isBarcode)
			{
				cv::Rect y_rt = it->roi;
				cv::rectangle(frame, y_rt, cv::Scalar(0, 255, 255), 2);
			}
		}

		list_barcode.clear();
	}

	cv::imshow("frame", frame);
	cv::waitKey(0);

	cv::destroyAllWindows();

	return 0;
}