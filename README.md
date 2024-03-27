# BarBeR: Barcode Benchmark Repository
The repository contains multiple algorithms for 1D and 2D barcode localization proposed in different papers in the past years. The repository contains the tools to measure the performance of those algorithms
![Barber Logo](./logo.png?raw=true)
# Installation Instructions
To build the libraries you first need to install:
- OpenCV for C++ (v4) and OpenCV-contrib (Instructions [here](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)) 
- Install Boost with the following command: sudo apt -y install libboost-filesystem-dev

Then, you can build the repository using CMake:
```
mkdir build
cd build
cmake ..
cmake --build .
```

# Available methods
* Gallo2011
* Tekin2012
* Soros2013
* Zamberletti2013
* Creusot2016
* Yun2017
