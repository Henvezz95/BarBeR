# Vision-based 1D barcode localization method for scale and rotation invariant

This is a barcode detection project at the digital signal processing laboratory of the Sungkyunkwan University. With the program in this repository you can evaluate your data. 

The software runs under windows and linux and is written in C++.

Based on the following libraries:
* OpenCV ( >= 3.1)
* CMake (>= 2.8)

Build Instructions
------------------
    $ git clone  https://github.com/iyyun/Barcode_1D.git
    $ cd Barcode_1D/src/Linux/tencon
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make

Dataset
-------
you can download our dataset from [here](http://dspl.skku.ac.kr/home_course/data/barcode/skku_inyong_DB.zip) (not include ground truth labels)


Demo
----
    $ cd Barcode_1D/src/Linux/tencon/build
    $ ./iyBarcode --file=../../tencone/Test_images/t1.jpg     

[![video](http://img.youtube.com/vi/KbB97vP3mhA/0.jpg)](https://youtu.be/KbB97vP3mhA?t=0s)

Cite
----

yon can find the paper in the "Document" section or [here](https://www.researchgate.net/publication/321349040_VIsion-based_1D_Barcode_Localization_Method_for_Scale_and_Rotation_Invariant) or [IEEEXelore](https://ieeexplore.ieee.org/document/8228227).

If you use this barcode detection program, please cite the following:

    @inproceedings{yun2017vision,
            title={Vision-based 1D barcode localization method for scale and rotation invariant},
            author={Yun, Inyong and Kim, Joongkyu},
            booktitle={Region 10 Conference, TENCON 2017-2017 IEEE},
            pages={2204--2208},
            year={2017},
            organization={IEEE}
        }
