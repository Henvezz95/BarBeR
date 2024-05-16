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

# Saved Models
One Saved Model for every architecture and scale can be downloaded form [here](https://1drv.ms/f/s!AhGbwgwB_qwFgbA0hLye0PkUnmYkVA?e=JMAX5e). Unzip the folder and place "Saved Models" directly inside the main repository folder.

# Compatibility
The repository has been developed with Linux as the main target OS. However, it should be possible to build the project also on Windows. The code is not architecture-specific and it's possible to build and run all the tests on different architectures. Both x86-64 and ARM architectures have been tested without any reported issues.

# Folders
* **algorithms**: Contains a Python class for every localization algorithm available. In particular, the available classes are:
  - detectron2_detector.py: loads a Detectron2 model in .pt or .pth format and uses it for localization.
  - gallo_detector.py: runs the 1D barcode localization method proposed by Orazio Gallo and Roberto Manduchi in the 2011 paper ["Reading 1D Barcodes with Mobile Phones Using Deformable Templates"](https://pubmed.ncbi.nlm.nih.gov/21173448/).
  - pytorch_detector.py: loads a Pytorch detection model in .pt or .pth format and uses it for localization.
  - soros_detector.py: runs the 1D and 2D barcode localization method proposed by G. Sörös and C. Flörkemeier in the 2013 paper [Blur-resistant joint 1D and 2D barcode localization for smartphones](https://dl.acm.org/doi/10.1145/2541831.2541844).
  - tekin_detector.py: runs the 1D barcode localization method proposed by E. Tekin et al. in the 2013 paper ["S-K Smartphone Barcode Reader for the Blind"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4288446/).
  - ultralytics_detector.py: loads an Ultralytics model (YOLO or RT-DETR supported) in .pt or .pth format and uses it for localization
  - yun_detector.py: runs the 1D barcode localization method proposed by I. Yun and K. Joongkyu in the 2017 paper ["Vision-based 1D barcode localization method for scale and rotation invariant"](https://ieeexplore.ieee.org/abstract/document/8228227).
  - zamberletti_detector.py: runs the 1D barcode localization method proposed by A. Zamberletti et al. in the 2013 paper ["Robust Angle Invariant 1D Barcode Detection"](http://artelab.dista.uninsubria.it/res/research/papers/2013/2013_zamberletti_acpr.pdf).
  - zharkov_detector.py: uses the deep-learning architecture proposed by A. Zharkov and I. Zagaynov in the 2019 paper ["Universal Barcode Detector via Semantic Segmentation"](https://arxiv.org/abs/1906.06281). The model must be a Pytorch model. The class can be used for both 1D and 2D barcode detection.
* **config**: contains the .yaml configuration files for each Python script that needs a configuration file. These configuration files are examples and can be modified depending on the configuration needed.
* **python**: contains all Python files, including all test scripts. In particular:
  - test_single_class.py: runs a set of detection algorithms on the test set considering only barcodes of one class i.e. 1D or 2D. The test measures precision, recall, F1-scores, mAP0.5, and mAP[0.5:0.95] of all methods.
  - test_multi_class.py: runs a set of detection algorithms on the test set considering all images. The test measures precision, recall, F1-scores, mAP0.5, and mAP[0.5:0.95] of all methods.
  - time_benchmark.py: runs a set of detection algorithms on the test set (or part of it) and measures the mean processing times of all methods.

* **scripts**: contains bash scripts to run pipelines of Python files (useful for k-fold cross-validation)
* **results**: contains the results produced by running the tests. Results are divided into 2 categories:
  - reports: are .yaml files generated after running a Python test
  - graphs: are .png files representing a graph generated using one or multiple .yaml reports
 
All other folders are needed to compile the necessary libraries when building the repository.
 
# Generate Train-Test Split Annotations
For a test we need COCO annotations divided into train.json, val.json, and test.json. To configure how to split the annotations, we use a configuration file. An example is ```config/generate_coco_annotations_config.yaml```. With the configuration file, we can select which files to use and which annotations, the train-test split size, and if we are using K-fold cross-validation.
The script used to generate the annotation is ```python/generate_coco_annotations.py```, which takes as input a configuration file and optionally the index k, which indicates the index of the current cross-validation test.

```
python3 python/generate_coco_annotations.py -c ./config/generate_coco_annotations_config.yaml  -k 0
```

If we also need to train an Ultralytics model, we need YOLO annotations, which will be generated with the following command:

```
python3 python/convert_coco_to_yolo.py -c ./annotations/COCO/ -o "./dataset/
```
