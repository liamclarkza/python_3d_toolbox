# Camera Toolbox Docs

## __Setting up a Python Environment__
### Conda
### Setuptools

## __1. Creating a Project__
In order to use the toolbox, you will need to import it
```python
# import the camera toolbox
import cameratoolbox as ctb
```

To create a new project, run the command below specifying the path of the project folder you wish to create. This will create the project folder and the necessary subdirectories for you. The command will return a path to the project directory which will be used later.
```python
# create a new project in a new folder named 
# "my_test_project" on the desktop
project_dir = ctb.create_new_project('~/Desktop/my_test_project')

```

## __2. Camera Calibration__
In order to calibrate the cameras, you will need to record videos of someone moving a calibration board (typically a checkerboard) infront of the cameras. 

In order to calibrate the extrinsic parameters (the translation and rotation between the cameras), you will need to ensure that each consecutive pair of cameras (i.e. 1-2, 2-3, 3-4...) can see the calibration board at the same time for several frames. 

Try to get atleast 20 pairs of frames from different views for each consecutive pair of cameras.

### __2.1. Finding calibration board corner points__
In order to calibrate the cameras, we first need to find frames in the videos which contain the calibration board and locate the points of the corners.

To do this you will need to call the ```find_corners``` function specifying the shape and the length of the square edges on the calibration board.

The calibration board's shape is determined by the number of internal corners. If a calibration board has *width x height* of *M x N*, it will have a shape of *(M-1) x (N-1)*.

The square edge length can simply be determined by measuring the length of one of the sides of the board's squares (in metres).

e.g. for a calibration board with shape 9x6 and a square edge length of 88mm, the following should be called:

```python
# finds corners located in the videos located in 
# <project_dir>/calibration directory
find_corners(project_dir, (9,6), 0.088)
```

### __2.2. Intrinsic calibration__
The cameras need to be calibrated using the function below. The camera model and resolution need to be specified. The camera model can either be *standard* or *fisheye*.
```python
# calibrates the cameras using the points found
# from find_corners which were saved in the
# <project_dir>/calibration directory
calibrate_intrinsics(project_dir, 'fisheye', (2704, 1520))
```
### __2.3. Extrinsic calibration__
To calibrate the extrinsic camera parameters, use the function below. This funciton can optionally output a 3D video of the calibration process allowing you to check for any obvious errors.
```python
# calibrates the extrinsic camera parameters using the points found
# from find_corners() and camera intrinsic parameters from 
# calibrate_intrinsics() which were saved in the
# <project_dir>/calibration directory
calibrate_extrinsics(project_dir, output_video_filepath='~/Desktop/test.mp4')
```

## __3. 2D to 3D Reconstruction__
At the moment, only I have only implemented a wrapper for parsing 2D data from DeepLabCut to 3D data. More coming soon
### __3.1 DeepLabCut__
To plot data from DLC in 3D, use the function below:
```python
files = [
    './07_03_2019MenyaRun1CAM1DLC_resnet50_CheetahOct14shuffle1_200000.h5',
    './07_03_2019MenyaRun1CAM2DLC_resnet50_CheetahOct14shuffle1_200000.h5',
    './07_03_2019MenyaRun1CAM3DLC_resnet50_CheetahOct14shuffle1_200000.h5',
    './07_03_2019MenyaRun1CAM4DLC_resnet50_CheetahOct14shuffle1_200000.h5',
    './07_03_2019MenyaRun1CAM5DLC_resnet50_CheetahOct14shuffle1_200000.h5',
    './07_03_2019MenyaRun1CAM6DLC_resnet50_CheetahOct14shuffle1_200000.h5',
    ]
dlc_to_3d(project_dir, files, output_video_filepath='/Users/liam/Desktop/testDLC.mp4')
```