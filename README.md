# BarBeR: Barcode Benchmark Repository
The repository contains multiple algorithms for 1D and 2D barcode localization proposed in different papers in the past years. The repository contains the tools to measure the performance of those algorithms


<img src='./logo.png' width='200'>

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

# Download the Dataset
The dataset can be downloaded from this [Link](https://unimore365-my.sharepoint.com/:f:/g/personal/319554_unimore_it/EpO-JIoN9HlJlvLBB4cZhREBTTiScfGMg6t1s68ifrtHMQ?e=gRHz0T).
Once Unzipped, you will find 2 folders inside: "Annotations" and "dataset". If you place these two folders directly inside the BarBeR folder there is no need to change the paths of the configuration files.
