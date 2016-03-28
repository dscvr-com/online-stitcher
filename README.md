# optonaut-online-stitcher
Dev Crib for Optonaut Recording and Aligning on Phones

## Input Data Format

An input data package consists of a number of data/image pairs (`NUMBER.json`/`NUMBER.jpg`).

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

An output data package consists of a zipped file ("opto-file") that contains a pair of JPEG images (`left.jpg`, `right.jpg`) and a data file, `data.json`.

The file `data.json` contains:
* `id`, UUID of this Oprograph. 
* `version`, the version of the stitcher that created the Optograph.
* `orientation`, the orientation of the center of the Optograph, given as 3x3 rotation matrix, in iOS core motion reference frame. 
* `alignment`, either `top`, `center` or `bottom`, specifying the texture alignment. 
* `offset`, optional, a pair of sperical coordinates, specifying the offset of the texture on the sphere. 
* `author`, optional, the name of the author. 
* `description`, optional, a description.
* `timestamp`, optional, the date of creation. 
* `hastags`, optional, list of hashtags associated with this Optograph. 

Example: 

```json
{
    "id": "550e8400-e29b-11d4-a716-446655440000",
    "version": "7.0.0",
    "orientation": [0.274960309267044, 0.0712836310267448, 0.958809375762939, 0, -0.152490735054016, 0.98785811662674, -0.0297131240367889, 0, -0.949285745620728, -0.138039633631706, 0.282491862773895, 0], 
    "alignment": "center", 
    "author": "schickling",
    "hashtags": ["test", "london"]
}

```

## Notes on compiling on Mac
Build Flags used for OpenCV
```
cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules -DWITH_OPENGL=ON -DWITH_OPENEXR=OFF -DBUILD_TIFF=OFF -DWITH_CUDA=OFF -DWITH_NVCUVID=OFF -DBUILD_PNG=OFF ..
```

Please set the environment variable OpenCV_DIR to the path of opencv. 

Please keep track of all the dependencies when compiling OpenCV. Ceres solver is especially important for SFM. 


## Some notes on the image-processing code structure: 
* Most code is placed in header files. This is due to compile-time optimization and ease-of-use. There are some exceptions with code that does not work with the iOS/Android build process. 
* There are some examples and tests in the root folder. Also, there are unit tests for some low level modules in the test folder. 
* The code consists roughly of the following modules: 
** debug - Contains debug code that is not used on the phone. 
** common - Contains several general classes and modules, mainly generic helpers for other modules. Example: Graph, Ring Buffers, Timers, other Collection Types. 
** imgproc - Contains classes that work on image data directly. For example alignment.  
** io - Contains classes for image IO, especially for storing contextual information during stitching. 
** math - Contains mathematical functions. Especially related to projection, quaternions and statistics. 
** recorder - Contains classes specific for recording. The main class here is Recorder. 
** stereo - Contains the code responsible for stereo conversion. 
** stitcher - Contains code responsible for stitching results together. The main class here is RingStitcher (and MultiRingStitcher). Also, classes for very simple debug stitching exist. 
** minimal - Contains code to use parts of this project in a very simple and minimal way. Great for testing!


