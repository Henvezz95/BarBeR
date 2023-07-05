# LCEBDFTB
Low-Computation Egocentric Barcode Detector for the Blind

Implementation of the algorithm proposed in:

C. Creusot and A. Munawar: Low-computation egocentric barcode detector for the blind. IEEE International Conference on Image Processing (ICIP), 2016

## Installation

### Building from Source

#### Dependencies

- [OpenCV 3.2](http://www.opencv.org) (Open Source Computer Vision Library)

#### Building

To build from source, clone the latest version from this repository and compile the package using

	git clone https://github.com/AndreasAZiegler/LCEBDFTB.git
	cd LCEBDFTB
  	mkdir build
 	cd buil
 	cmake ..
	make
  	make install

After this, you will find the library in the folder "lib" and the headers in the folder "include".
