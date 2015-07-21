# optonaut-online-stitcher
Dev Crib for Optonaut Recording and Aligning on Phones

## Data Format

A data package consists of a number of data/image pairs (`NUMBER.json`/`NUMBER.jpg`).

* `NUMBER.json` includes the `intrinsics` matrix (3x3) of the camera, the `extrinsics` matrix (4x4) of the respective frame in row format and an integer `id` which is the same as `NUMBER`.

  ```json
  {
    "id": 5,
    "intrinsics": [4.854369, 0, 3, 0, 4.854369, 2.4, 0, 0, 1],
    "extrinsics": [0.274960309267044, 0.0712836310267448, 0.958809375762939, 0, -0.152490735054016, 0.98785811662674, -0.0297131240367889, 0, -0.949285745620728, -0.138039633631706, 0.282491862773895, 0, 0, 0, 0, 1]
  }
  ```
  
* `NUMBER.jpg` is the image of the respective frame.
