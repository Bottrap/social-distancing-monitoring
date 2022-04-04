# Social Distancing Monitoring using MATLAB

## Introduction
The main purpose of our project is to propose 3 existing approaches, based on research projects, to estimate social distance, detecting people in the image and displaying the corresponding bird's eye view.

![result example](https://raw.githubusercontent.com/Bottrap/social-distancing-monitoring/main/img_example/result_example.png)

## Getting Started
### Requirements
The code requires the following toolboxes to be installed:
- Image Processing Toolbox
- Computer Vision Toolbox
- Robotics System Toolbox
- Mapping Toolbox
- Deep Learning Toolbox

The code also requires the following support package to be installed:
- Deep Learning Toolbox Converter for ONNX Model
### Project Directory
The project directory appears structured as follows. The folders utils (in openpose folder) and yolov4 are not detailed because contain only files and other folders which are dependencies necessary for the execution.

```sh
${project_dir}/
├── img_example
│   ├── draw_polygon_example.png
│   └── result_example.png
├── dataset
├── meter_per_pixel
│   ├── korte.m
│   └── mall.m
├── openpose
│   ├── utils
│   └── openpose_korte.m
├── oxford_camera_params
│   ├── oxford_BEV_draw.m
│   └── oxford_BEV_fissa.m
├── yolov4
├── result_example.png
└── utils.m
```

### Download Dataset
Download the dataset folder from this link -> [download dataset](https://www.mediafire.com/file/qxhyr40vgyjmz3f/dataset.rar/file)

Move the dataset folder in the project dir directly so that it appears as explained in the [project directory section](#project-directory).

KORTE's dataset is composed of images from two scenarios.

**Scenario 1 example image**
![scenario 1 image example](https://github.com/Bottrap/social-distancing-monitoring/blob/main/img_example/scenario_1_example.JPG)

**Scenario 2 example image**
![scenario 2 image example](https://github.com/Bottrap/social-distancing-monitoring/blob/main/img_example/scenario_2_example.JPG)

## Run Social Distancing Monitoring using Camera Parameters

In the case you are using our dataset, you don’t have to change anything, except the limit of frameNumber if you want to watch different video scene.

In other case, select your video and select your frame limit (currently set to 30).
```MATLAB
videoReader = VideoReader("../dataset/TownCentreXVID.avi");
frameNumber = 0;
while frameNumber < 30
    frame = readFrame(videoReader);
    frameNumber = frameNumber + 1;
end
I = readFrame(videoReader);
```
If you are using other video or image, you have to set the camera parameters: intrinsic (focal length, optical center, lens distorsion) and extrinsic (rotation and translation).
```MATLAB
% Camera parameters about Oxford dataset
F_X = 2696.35888671875000000000;
F_Y = 2696.35888671875000000000;
C_X = 959.50000000000000000000;
C_Y = 539.50000000000000000000;
S = 0;
quaternion = [0.49527896681027261394 0.69724917918208628720 -0.43029624469563848566 0.28876888503799524877];
translationVector = [-0.05988363921642303467 3.83331298828125000000 12.39112186431884765625];
```
### Run oxford_BEV_draw.m
When you run the oxford_BEV_draw.m file you have to draw a polygon to define the region of interest
![draw polygon example](https://raw.githubusercontent.com/Bottrap/social-distancing-monitoring/main/img_example/draw_polygon_example.png)
### Run oxford_BEV_fixed.m
When you run the oxford_BEV_fixed.m file the algorithm define for you a fixed region of interest to be a rectangle in the bird's eye view.

## Run Social Distancing Monitoring using Meter per pixel method
Before running the script, both for korte.m and for mall.m you can choose the detector to use:
- yolo = false -> use the Matlab peopleDetectorACF()
- yolo = true -> use YOLOv4-coco
```MATLAB
yolo = false;
```
In order to use this method with korte dataset (script korte.m) it is necessary to select an area of 4x4 floor tiles if using an image of the first scenario or an area of 4x2 floor tiles (since the short side of the tile is the half of the long side) if using an image of the second scenario, as shown in the project report.

## Run Social Distancing Monitoring using OpenPose method
You can choose the image among those in the dataset but if you change the camera and use your own image, you have to change also the sensor height and the sensor width.
```MATLAB
imgPath = '../dataset/KORTE/data/_MG_8781.JPG';
sensor_height = 24; %mm
sensor_width = 36; %mm
```
Note that the OpenPose library represents the bottleneck of this system as it is not always very accurate in the detections.
![bad OpenPose detection example](https://github.com/Bottrap/social-distancing-monitoring/blob/main/img_example/openpose_bad_detections.jpeg)
