# DSCVR Image Stitcher

Repository of DSCVR's (former Optonaut's) image processing funcitonality. The goal of this project is to provide stereo panorama stitching on a mobile phone. 

The code here can be cross-compiled for iOS, Android and Linux. The main dependency is the OpenCV framework. 

When the project is built using the supplied cmake script, a library, as well as some test applications are generated. Those can be used to test and develop the image stitcher on a PC. 

## Usage

For useage examples, please look at the main testing tool (`src/stitcherTest.cpp`) as well as the iOS and Android applications. 

## How does it work

To accomodate the recording process on smarthones, the workflow is roughly the following: 

1. Creation of a Recorder instance, which will hold our state for recording. There are different Recorder instances for single ring (`Recorder2`) and multi-ring recording (`MultiRingRecorder2`). The areason is that plenty of performance optimizations can be done for single ring recording due to smaller memory requirements. 
2. Creation of a `RecorderGraph` instance which represents a series of images that need to be recorded. It s important to choose appropriate overlaps between images. For stereo panoramas, this overlap is typically greater than for regular panormas. For this step, we need at least rough knowledge of the intrinsic parameters of the device's camera, e.g. focal length and sensor size.
3. Images are supplied to the Recorder's push method. This will update the Recorder's internal state and save the image for later, if applicable. Since the state is updated, the values returned by `GetBallPosition()` and `GetDistanceToBall()` will be up to date and can be used to show a hint in the UI of the application. Each image saved for stitching will be compared to the neighboring image, for later position and exposure correction. 
4. After recording is finished, the results of the pairwise comparison of images will be used to do a simplified version of global bundle adjustment. This way, we receive an estimate of the camera's focal length and each image's position. 
5. Each pair of neighboring images is not re-projected on a plane between the two images. The plane is tangential to the sphere all images (and also our result panorama) lie upon. From the re-projected images, we cut equal-sized pieces, which we now use for stitching the panorama.
6. The panorama stitching itself uses flow-based blending. For each overlapping region, an optical flow is calculated. We fuse neighboring images by blending pixel position and color linearly. 

This paper gives a great summary of a very similar workflow, altough it is not related to this project and was published after this project was developed. 

## Project Structure

Most code is placed in header files. This is due to compile-time optimization and ease-of-use. There are some exceptions with code that does not work with the iOS/Android build process. 
There are some examples and tests in the ```src``` folder. Also, there are unit tests for some low level modules in the ```src/test``` folder. 

The code consists roughly of the following modules: 
* debug - Contains debug code that is not used on the phone. 
* common - Contains several general classes and modules, mainly generic helpers for other modules. Example: Graph, Ring Buffers, Timers, other Collection Types. 
* imgproc - Contains classes that work on image data directly. For example alignment.  
* io - Contains classes for image IO, especially for storing contextual information during stitching. 
* math - Contains mathematical functions. Especially related to projection, quaternions and statistics. 
* recorder - Contains classes specific for recording. The main class here is Recorder. 
* stereo - Contains the code responsible for stereo conversion. 
* stitcher - Contains code responsible for stitching results together. The main class here is RingStitcher (and MultiRingStitcher). Also, classes for very simple debug stitching exist. 
* minimal - Contains code to use parts of this project in a very simple and minimal way. Great for testing!

## Input Data Format

An input data package for the test applications consists of a number of data/image pairs (`NUMBER.json`/`NUMBER.jpg`).

* `NUMBER.json` includes the `intrinsics` matrix (3x3) of the camera, the `extrinsics` matrix (4x4) of the respective frame in row format and an integer `id` which is the same as `NUMBER`.

  ```json
  {
    "id": 5,
    "intrinsics": [4.854369, 0, 3, 0, 4.854369, 2.4, 0, 0, 1],
    "extrinsics": [0.274960309267044, 0.0712836310267448, 0.958809375762939, 0, -0.152490735054016, 0.98785811662674, -0.0297131240367889, 0, -0.949285745620728, -0.138039633631706, 0.282491862773895, 0, 0, 0, 0, 1]
  }
  ```
  
* `NUMBER.jpg` is the image of the respective frame.

## Output Data Format

The output data format depends on the tool used. For the main test application, it's usually a pair of panorama images, one for each eye. 

## Test Programs

The test programs in the main directory have the following functions:
* alignerTest - testbed for testing pixel-based alignment functions on two plain images. Usage: aligner-test image1 image2
* bruteForceAlignerTest - testbed for testing iterative bundle adjustment on a collection of images + metadata. Usage: brute-force-aligner-test inputimages
* perspectiveAlignerTest - testbed for testing pixel-baded alignment on two images + metadata. The images are transformed accordingly prior alignment. Usage: perspective-aligner-test image1 image2
* ringClosureTest - testbed for testing closing of first ring. Usage: ring-closure-test inputImages
* stereoConversionTest - testbed for testing stereo conversion with two images + metadata. Usage: stereo-conversion-test image1 image2
* stitcherTest/optonautTest - testbed for whole recording/stitching pipeline on a collection of images + metadata. Usage: optonaut-test inputImages

All test programs except stitcherTest support the following command line arguments: 
* -s [NUMBER] skips NUMBER images between each input image. 
* -l [NUMBER] limits the processed input images to NUMBER
* -m [MODE] Set MODE to ```a``` for Android, ```n``` for no conversion. IOS is default. 

For changing the mode of stitcherTest, please edit the ```mode``` constant in stitcherTest.cpp.

## Utility Programs

The utility programs in the main directory are small stand-alone tools that are used around the whole optonaut system. They have the following functions: 
* panoBlur - extens a single-ring panorama by a blurrend and mirrored area, so it looks more pleasing in VR. Usage: pano-blur inputImage outputImage
* toCubeMap - converts a equirectangular panorama to it's cube map representation. Usage: to-cube-map [INPUT-IMAGE] [OUTPUT-IMAGE] [WIDTH] [FACE-ID] [SUB X] [SUB Y] [SUB WIDTH] [SUB HEIGHT]
* toPol - converts a equirectangular panorama to it's inverse polar projection (e.g. little world). Usage: to-polar inputImage outputImage

## Experimental Code

The follwoing code files are experimental: 
* debugHookTest - test of 3D-based debugging tool, that shows images/features in a 3D space. 
* featureChainTest, minimalFeatureTest - testbeds for feature based alignments and structure from motion, e.g. creating 3D models from a series of images. 
* stereoMatchTest - testbed for stereo matching, e.g. comparing two images to extract a depthmap. 

## Matrix conventions

Unless otherwise noted, we use ```CV_64F``` (```double```) matrices for all matrices in mathematicel sense (also quaternions and vectors). We use ```CV_8UC3``` matrices for BGR images and ```CV_8UC1```matrices for grayscale images. 

All intrinsic matrices are represented as 3x3 matrices, all extrinsic matrices are represented as 4x4, even if they only contain rotations. 

## Compiling 

Cmake and a c++13 compiler are required. For example ```clang 7.2.0``` or ```gcc 4.9```. 

OpenCV is the main dependency. Build Flags used for OpenCV:
```
cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -DWITH_OPENGL=ON -DWITH_OPENEXR=OFF -DBUILD_TIFF=OFF -DWITH_CUDA=OFF -DWITH_NVCUVID=OFF -DBUILD_PNG=OFF ..
```
The contrib modules are used for the experimental SFM support. 

Please set the environment variable OpenCV_DIR to the path of opencv. 

Please keep track of all the dependencies when compiling OpenCV. Ceres solver is especially important for SFM. 


## Some notes on the image-processing code structure: 
* Most code is placed in header files. This is due to compile-time optimization and ease-of-use. There are some exceptions with code that does not work with the iOS/Android build process. 
* There are some examples and tests in the root folder. Also, there are unit tests for some low level modules in the test folder. 

The code consists roughly of the following modules: 
* debug - Contains debug code that is not used on the phone. 
* common - Contains several general classes and modules, mainly generic helpers for other modules. Example: Graph, Ring Buffers, Timers, other Collection Types. 
* imgproc - Contains classes that work on image data directly. For example alignment.  
* io - Contains classes for image IO, especially for storing contextual information during stitching. 
* math - Contains mathematical functions. Especially related to projection, quaternions and statistics. 
* recorder - Contains classes specific for recording. The main class here is Recorder. 
* stereo - Contains the code responsible for stereo conversion. 
* stitcher - Contains code responsible for stitching results together. The main class here is RingStitcher (and MultiRingStitcher). Also, classes for very simple debug stitching exist. 
* minimal - Contains code to use parts of this project in a very simple and minimal way. Great for testing!


Please keep track of all the dependencies when compiling OpenCV. Ceres solver and Eigen are only important for SFM. 
The visual debug hook and the SFM preview depends on the 3D library irrlicht. 

The code can be compiled by running ```./compile.sh``` in the root directory. 

## Hardware Accellerated debugging

Hardware accellerated computation uses GLSL with SFML. 
This is natively supported on almost all OS. 

If you need to run it on Bash for Windows, install an XServer on windows (VcXsrv), invoke it with the command line (so the window is not hidden), then in Bash export the disaply (export DISPLAY=:0). 

This will create a fine rendering context. 

## Parameter Tuning

The most tuneable global params are hOverlap/vOverlap for recorderGraphGenerator, which gives how many images we record, as well as hOverlap/vOverlap for StereoGenerator, which gives how much additional overlap we add to the sides of the image for later stitching. Changing these parameters might be necassary for accomodating different camera or phone types. 
