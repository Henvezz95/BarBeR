# BLaDE

> This library is based on the reference implementation of "BLaDE: Barcode Localization and Decoding Engine" [[1]](http://www.ski.org/blade-barcode-localization-and-decoding-engine-0) which can be downloaded at <https://sourceforge.net/projects/sk-blade/>.

Barcode Localization and Decoding Engine (BLaDE) is aimed at developing an open-source 1-D barcode localization and decoding engine, aimed primarily at blind and visually impaired users. BLaDE can detect and decode 1-D barcodes at any orientation, as well as detect partial barcodes and provide real-time audio feedback to help the user orient a smartphone camera or webcam to optimally resolve the barcode.

## Building for Android

```
ndk-build NDK_APPLICATION_MK=DroidBLaDE/jni/Application.mk NDK_PROJECT_PATH=DroidBLaDE/
```

## Original README content

Source files for the BLaDE library, as well as the linux and android clients.

Compilation:
The simplest way is the check out the whole repo into a new Eclipse workspace, and import the projects into the existing workspace. Then, build BLaDE, LinBLaDE and DroidBLaDE. LinBlaDE depends on the opencv versions of the BLaDE libraries, which are placed into the /lib directory (which is created if it does not exist), and the necessary symbolic links are created. The /include directory contains some common headers for SKI projects that are utilized here.

The source code is released under a **BSD license**, please refer to the file SKI_BSD_LICENSE.

