# optonaut-online-stitcher
Dev Crib for Optonaut Recording and Aligning on Phones

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
* stitcherTest - testbed for whole recording/stitching pipeline on a collection of images + metadata. Usage: optonaut-test inputImages

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

## Notes on compiling on Mac

Build Flags used for OpenCV
```
cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -DWITH_OPENGL=ON -DWITH_OPENEXR=OFF -DBUILD_TIFF=OFF -DWITH_CUDA=OFF -DWITH_NVCUVID=OFF -DBUILD_PNG=OFF ..
```

Please set the environment variable OpenCV_DIR to the path of opencv. 

Please keep track of all the dependencies when compiling OpenCV. Ceres solver and Eigen are especially important for SFM. 

The visual debug hook and the SFM preview depends on the 3D library irrlicht. 
