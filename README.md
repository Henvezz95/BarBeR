# BarBeR: Barcode Benchmark Repository
The repository contains multiple algorithms for 1D and 2D barcode localization proposed in different papers in the past years. The repository contains the tools to measure the performance of those algorithms


<img src='./logo_with_name.png'>

## Publications & Reproducibility
> *Are you searching for *BaFaLo* 🦬, our proposed method for fast segmentation of barcodes?*&nbsp;Check it out **[Here](./BaFaLo)** !

| Year | Reference | Focus | Links |
|------|-----------|-------|-------|
| 2025 | **Vezzali _et_ _al._** “State-of-the-Art Review and Benchmarking of Barcode Localization Methods,” *Eng. Appl. of AI* | Complete description of the dataset, benchmarking tools, protocols, and an extensive method survey. **Primary reference.** | 📄 [Paper](https://www.sciencedirect.com/science/article/pii/S0952197625002593)|
| 2024 | **Vezzali _et_ _al._** “BarBeR: A Barcode Benchmarking Repository,” *Proc. ICPR* | Original dataset introduction & baseline results. | 📄 [Paper](https://link.springer.com/chapter/10.1007/978-3-031-78447-7_13) |
| 2024 | **Vezzali _et_ _al._** “BarBeR – Implementation and Reproducibility Notes,” *RRPR Workshop* | Step-by-step scripts, configs, and practical tips to reproduce our ICPR numbers on your hardware. | 📄 [Paper](https://iris.unimore.it/retrieve/daaf2bf0-5171-456e-bada-a273df0c6bb4/BarBeR___Barcode_Benchmark_Repository__Implementation_and_Reproducibility_Notes.pdf)|

> *Need the exact BibTeX?*&nbsp;See the **[How to Cite BarBeR](#how-to-cite-barber)** section at the end of this README.

# Installation Instructions
To build the libraries, you first need to install:
- OpenCV for C++ (v4) and OpenCV-contrib (Instructions [here](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)) 
- Install Boost with the following command: sudo apt -y install libboost-filesystem-dev (Ubuntu). For Windows, follow [these instructions](https://robots.uc3m.es/installation-guides/install-boost.html#install-boost-ubuntu).

Then, you can build the repository using CMake:
```
mkdir build
cd build
cmake ..
cmake --build .
```
To install all required Python libraries run this command:
```
pip install -r requirements.txt
```

# Download the Dataset
The dataset can be downloaded from this [Link](https://ditto.ing.unimore.it/barber/).
Once unzipped, you will find 2 folders inside: "Annotations" and "dataset". If you place these two folders directly inside the BarBeR folder, there is no need to change the paths of the configuration files. From the same link is also possible to download the pre-trained detection models.

<img src='./Examples.jpg' width='500'>

# Saved Models
One Saved Model for every architecture and scale can be downloaded from [here](https://ditto.ing.unimore.it/barber/). Unzip the folder and place "Saved Models" directly inside the main repository folder.

Additional segmentation models can be downloaded from [here](https://unimore365-my.sharepoint.com/:u:/g/personal/319554_unimore_it/EYz0mZdleahDoXHozIqgiF4BGUg-ppbonS3v9MVxd-FFcQ?e=2ug9O0). In the future, it will be possible to download everything from a single link.

# Compatibility
The repository has been developed with Linux as the main target OS. However, it should be possible to build the project also on Windows. The code is not architecture-specific, and it's possible to build and run all the tests on different architectures. Both x86-64 and ARM architectures have been tested without any reported issues.

# Folders
* **algorithms/detectors**: Contains a Python class for every localization algorithm available. In particular, the available classes are:
  - detectron2_detector.py: loads a Detectron2 model in .pt or .pth format and uses it for localization.
  - gallo_detector.py: runs the 1D barcode localization method proposed by Orazio Gallo and Roberto Manduchi in the 2011 paper ["Reading 1D Barcodes with Mobile Phones Using Deformable Templates"](https://pubmed.ncbi.nlm.nih.gov/21173448/).
  - pytorch_detector.py: loads a Pytorch detection model in .pt or .pth format and uses it for localization.
  - soros_detector.py: runs the 1D and 2D barcode localization method proposed by G. Sörös and C. Flörkemeier in the 2013 paper [Blur-resistant joint 1D and 2D barcode localization for smartphones](https://dl.acm.org/doi/10.1145/2541831.2541844).
  - tekin_detector.py: runs the 1D barcode localization method proposed by E. Tekin et al. in the 2013 paper ["S-K Smartphone Barcode Reader for the Blind"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4288446/).
  - ultralytics_detector.py: loads an Ultralytics model (YOLO or RT-DETR supported) in .pt or .pth format and uses it for localization
  - yun_detector.py: runs the 1D barcode localization method proposed by I. Yun and K. Joongkyu in the 2017 paper ["Vision-based 1D barcode localization method for scale and rotation invariant"](https://ieeexplore.ieee.org/abstract/document/8228227).
  - zamberletti_detector.py: runs the 1D barcode localization method proposed by A. Zamberletti et al. in the 2013 paper ["Robust Angle Invariant 1D Barcode Detection"](http://artelab.dista.uninsubria.it/res/research/papers/2013/2013_zamberletti_acpr.pdf).
  - zharkov_detector.py: uses the deep-learning architecture proposed by A. Zharkov and I. Zagaynov in the 2019 paper ["Universal Barcode Detector via Semantic Segmentation"](https://arxiv.org/abs/1906.06281). The model must be a PyTorch model. The class can be used for both 1D and 2D barcode detection.
* **algorithms/readers**: contains algorithms to decode barcodes from images. Usually are pipelines include a localization step and a decoding step. For this reason, one of the inputs is often a localizer from `algorithms/detectors`.
* **config**: contains the .yaml configuration files for each Python script that needs a configuration file. These configuration files are examples and can be modified depending on the configuration needed.
* **python**: contains all Python files, including all test scripts. In particular:
  - test_single_class.py: runs a set of detection algorithms on the test set, considering only barcodes of one class i.e., 1D or 2D. The test measures precision, recall, F1-scores, mAP0.5, and mAP[0.5:0.95] of all methods.
  - test_multi_class.py: runs a set of detection algorithms on the test set, considering all images. The test measures precision, recall, F1-scores, mAP0.5, and mAP[0.5:0.95] of all methods.
  - time_benchmark.py: runs a set of detection algorithms on the test set (or part of it) and measures the mean processing times of all methods.

* **scripts**: contains bash scripts to run pipelines of Python files (useful for k-fold cross-validation)
* **results**: contains the results produced by running the tests. Results are divided into 2 categories:
  - reports: are .yaml files generated after running a Python test
  - graphs: are .png files representing a graph generated using one or multiple .yaml reports
 
* **BaFaLo**: Here are the necessary Python files to define the BaFaLo architecture in PyTorch and train it. In addition, it is possible to train the segmentation models used as comparison and the BaFaLo variations used in the ablation studies.
 
All other folders are needed to compile the necessary libraries when building the repository.
 
# Generate Train-Test Split Annotations
For a test, we need COCO annotations divided into train.json, val.json, and test.json. To configure how to split the annotations, we use a configuration file. An example is ```config/generate_coco_annotations_config.yaml```. With the configuration file, we can select which files to use and which annotations, the train-test split size, and if we are using K-fold cross-validation.
The script used to generate the annotation is ```python/generate_coco_annotations.py```, which takes as input a configuration file and optionally the index k, which indicates the index of the current cross-validation test.

```
python3 python/generate_coco_annotations.py -c ./config/generate_coco_annotations_config.yaml  -k 0
```

If we also need to train an Ultralytics model, we need YOLO annotations, which will be generated with the following command:

```
python3 python/convert_coco_to_yolo.py -c ./annotations/COCO/ -o "./dataset/
```

# Benchmark Tests

### Single-Class Detection

Evaluates localization performance on images that contain **only 1 barcode class at a time** (either 1D *or* 2D).

| Script | `python/test_single_class.py` |
| --- | --- |
| **Required args** | `-c` config file • `-o` output folder |
| **Example configs** | • `./config/test1D_singleROI.yaml`  (single 1D barcode)  <br>• `./config/test1D_multiROI.yaml`  (multiple 1D barcodes)  <br>• `./config/test2D_singleROI.yaml`  (single 2D barcode) |

```bash
python3 python/test_single_class.py   -c ./config/test1D_singleROI.yaml   -o ./results/reports/test1D_singleROI_640
```

---

### Multi-Class Detection

Runs on the **full dataset (1D + 2D)** and reports detection *and* classification metrics.

| Script | `python/test_multi_class.py` |
| --- | --- |
| **Required args** | `-c` config file • `-o` output folder |
| **Example config** | `./config/test_multiclass.yaml` |

```bash
python3 python/test_multi_class.py   -c ./config/test_multiclass.yaml   -o ./results/reports/test_multiclass_640
```

---

### Time Benchmark

Measures average **inference time per image** for each localization method across the entire dataset (train + val + test).

| Script | `python/time_benchmark.py` |
| --- | --- |
| **Required args** | `-c` config file • `-o` output folder |
| **Example config** | `./config/timing_config.yaml` |

```bash
python3 python/time_benchmark.py   -c ./config/timing_config.yaml   -o ./results/reports/test_time_640
```

---

### Decoding‑Rate Test

End‑to‑end evaluation of **detection → cropping → decoding** using `pyzbar`.  
Reports the percentage of barcodes successfully decoded, divided by category (1D *or* 2D).

| Script | `python/test_decodings.py` |
| --- | --- |
| **Required args** | `-c` config file • `-o` output folder |
| **Example config** | `./config/decoding_testmulticlass.yaml` |

```bash
python3 python/test_decodings.py   -c ./config/decoding_testmulticlass.yaml   -o ./results/reports/test_decodings_320-1280
```

> **Tip:** All YAML files share the same structure—define the list of algorithms, preprocessing (e.g., resize resolution), and dataset split. Use the provided examples as templates for custom experiments.


# Draw Graphs from the results
To generate a graph from the results generated by a Single-Class Detection Test run ```python3 python/visualizer/single_class_graphs.py```.
To generate a graph from the results generated by a Multi-Class Detection Test run ```python3 python/visualizer/multi_class_graphs.py```.

To change the path of the input reports, change the variable 'base_path' present in both scripts. In the case of a single class detection test, it's necessary to select the right barcode type, changing the variable type, which could be '1D' or '2D'. Graphs will be generated in the folder ```results/graphs```.

# Train a deep-learning model
To train a model with Ultralytics run ```python/ultralytics_trainer.py```. A configuration file is needed (e.g. ```config/ultralytics_training_config.yaml```), as well as an output path for the trained model (Default is Saved Models).
</br></br>
To train a model with Detectron2 run ```python/detectron2_trainer.py```. A configuration file is needed (e.g. ```config/detectron2_training_config.yaml```), as well as an output path for the trained model (Default is Saved Models).
</br></br>
To train a Zharkov model, run ```Zharkov2019/zharkov_trainer.py```. A configuration file is needed (e.g. ```config/zharkov_training_config.yaml```), as well as an output path for the trained model (Default is Saved Models).

# K-fold cross-validation
To run K-fold cross-validation, it would be necessary to run the scripts multiple times manually. Since running the scripts multiple times and changing the configuration each time would take too much time, it is possible to automate the process using a bash script. The python file ```python/create_configuration_yaml.py``` is used to generate a new configuration each time, so to change the settings of a K-fold cross-validation test, the file ```python/create_configuration_yaml.py``` must be changed accordingly. Scripts can also be used to train multiple networks, each with a different test set.

Example train 5 Ultralytics networks:
```
source scripts/k_fold_training_ult.sh
```

Example train 5 Detectron2 networks:
```
source scripts/k_fold_training_det.sh
```

Example train 5 Zharkov networks:
```
source scripts/k_fold_training_zharkov.sh
```

Example run 5 single-class tests with 1D barcodes:
```
source scripts/k_fold_test_1D_singleROI.sh
```

Example run 5 single-class tests with 2D barcodes:
```
source scripts/k_fold_test_2D_singleROI.sh
```

Example run 5 multi-class tests:
```
source scripts/k_fold_test_multiclass.sh
```

# Testing a New Localization Algorithm
* The localization method must be defined inside a new Python file (e.g. ```new_algorithm.py```) and the file must be placed inside the ```algorithms/detectors``` folder
* Define a class with the implementation of the algorithm. To ensure compatibility, the new class should inherit from the abstract class "BaseDetector" defined in ```algorithms/detectors_abs.py```
* A detector must have at least these two methods: detect and get_timing
* **detect works** on a single image and outputs a list of detected bounding boxes, a list with the classes of the detections, and a list of confidence scores (between 0 and 1 if available, otherwise None)
* **get_timing** returns the processing time of the last detection in milliseconds. The use of ```perf_counter_ns``` is advised, because it has a [high resolution](https://peps.python.org/pep-0564/#annex-clocks-resolution-in-python) (around 100ns) on both Linux and Windows. The output of ```perf_counter_ns``` should then be divided by 1e6.
  
 ```python
# Defining the new class inside algorithms/detectors/new_algorithm.py
from detectors_abs import BaseDetector

class NewDetector(BaseDetector):
  def __init__():
    ...
  def detect(self, img):
    ...
  def get_timing(self):
    ...
```

* To enable the new algorithm in a test, it should be added to the algorithms list in the configuration file used in the test. Check the available configuration files in the Repository for the exact syntax required

---

## How to Cite BarBeR

If you use the BarBeR dataset, benchmark tools, or pre-trained models, please cite **at least the journal article** listed below.  
When space allows, we kindly encourage citing **both** publications, as they reflect complementary aspects of the project:

| Paper | When to cite |
| ----- | ------------ |
| **Vezzali, Enrico, et al. "State-of-the-art review and benchmarking of barcode localization methods." Engineering Applications of Artificial Intelligence** | The most complete and up-to-date description of the dataset, benchmark tools, evaluation protocols, and a thorough review of barcode localization methods. Use this as the **primary citation**. |
| **Vezzali, Enrico, et al. "Barber: A barcode benchmarking repository." International Conference on Pattern Recognition. Springer, Cham, 2025.** | The original introduction of the BarBeR dataset and benchmark, including baseline results. Consider citing this **in addition** when discussing dataset construction or reproducing the original experiments. |

### BibTeX

```bibtex
@article{vezzali2025state,
  author    = {Vezzali, Enrico and Bolelli, Federico and Santi, Stefano and Grana, Costantino},
  title     = {{State-of-the-art Review and Benchmarking of Barcode Localization Methods}},
  journal   = {{Engineering Applications of Artificial Intelligence}},
  year      = {2025},
  volume    = {},
  pages     = {1--29},
  issn      = {0952-1976}
}

@inproceedings{vezzali2024barber,
  title={Barber: A barcode benchmarking repository},
  author={Vezzali, Enrico and Bolelli, Federico and Santi, Stefano and Grana, Costantino},
  booktitle={International Conference on Pattern Recognition},
  pages={187--203},
  year={2025},
  organization={Springer}
}
