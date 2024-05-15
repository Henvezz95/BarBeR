# BarBeR: Barcode Benchmark Repository
The repository contains multiple algorithms for 1D and 2D barcode localization proposed in different papers in the past years. The repository contains the tools to measure the performance of those algorithms


<img src='./logo.png' width='200'>

# Installation Instructions
To build the libraries you first need to install:
- OpenCV for C++ (v4) and OpenCV-contrib (Instructions [here](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)) 
- Install Boost with the following command: sudo apt -y install libboost-filesystem-dev (ubuntu). For windows, follow [these instructions](https://robots.uc3m.es/installation-guides/install-boost.html#install-boost-ubuntu).

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

# Folders
* algorithms: Contains a Python class for every localization algorithm available. In particular, the available classes are:
  - detectron2_detector.py: loads a Detectron2 model in .pt or .pth format and uses it for localization
  - gallo_detector.py: runs the 1D barcode localization method proposed by Orazio Gallo and Roberto Manduchi in the 2011 paper ["Reading 1D Barcodes with Mobile Phones Using Deformable Templates"](https://pubmed.ncbi.nlm.nih.gov/21173448/)
  - pytorch_detector.py: loads a Pytorch detection model in .pt or .pth format and uses it for localization
  - soros_detector.py: runs the 1D and 2D barcode localization method proposed by G. Sörös and C. Flörkemeier in the 2013 paper [Blur-resistant joint 1D and 2D barcode localization for smartphones](https://dl.acm.org/doi/10.1145/2541831.2541844)
  - tekin_detector.py:
  - ultralytics_detector.py: loads an Ultralytics model (YOLO or RT-DETR supported) in .pt or .pth format and uses it for localization
  - yun_detector.py:
  - zamberletti_detector.py:
  - zharkov_detector.py: uses the deep-learning architecture proposed by A. Zharkov and I. Zagaynov in the 2019 paper ["Universal Barcode Detector via Semantic Segmentation"](https://arxiv.org/abs/1906.06281). The model must be a pytorch model. The class can be used for both 1D and 2D barcode detection.
* python: contains all Python files, including all test scripts. In particular:
  -
  -
* scripts: contains bash scripts to run pipelines of python files (useful for k-fold cross-validation)
* results: contains the results produced by running the tests. Results are divided into 2 categories:
  - reports: are .yaml files generated after running a Python test
  - graphs: are .png files representing a graph generated using one or multiple .yaml reports

All other folders are needed to compile the necessary libraries when building the repository.
