# **CUSTOM MULTI POSE ESTIMATION WITH Lightweight OpenPose**

**-->custom_pose folder**



####                           **(always adjust the paths ad check them)**



**I have taken the starting code here <https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch>.** 

The repository of the link above contains the code (in Pytorch) for train and do inference for the human body with its 18 related keypoints.

requsites (from the repo):

torch==0.4.1

torchvision==0.2.1

pycocotools==2.0.0

opencv-python==3.4.0.14

numpy==1.14.0

## Create the dataset:

We need to create a dataset wich is compatible to the way the code is implemented, in this case we need a COCO dataset format (.json).

To do that I used this application running on docker <https://github.com/jsbroks/coco-annotator> (a bit tricky on Windows).

Follow the instructions displayed in the repo to run it.

Anyway download the repo and navigate to the root directory of the cloned repository and run the Docker command for production build:

```
$ docker-compose up
```

This command will pull the latest stable image from [dockerhub](https://hub.docker.com/r/jsbroks/coco-annotator), which pre-complied with all dependencies. The instance can be found at `http://localhost:5000/`. 

https://github.com/mathblue/Custom-Pose-Estimation-Yaw-Roll-Pitch-detection-tensorrt-compilation/blob/master/doc-img/docker-usage.PNG?raw=true

To import the images:

`cp /Users/Utente/Desktop/annotator/coco-annotator-master/datasets/<localdirectory>/ annotator_webclient:/datasets/<docker dataset folder>`

![cat2](C:\Users\Utente\Desktop\POSE-ESTIMATION\doc-img\cat2.PNG)

In this last image:

A: you need to add some keypoints labels categories.

B:put the keypoint where you want.

C: draw a polygon around the item, save, and go to the next image.

When you have finished labelling you can easily export coco with a button in the app and download the .json file.

I suggest to resize the images before training and labelling to a small size.

## Training (GPU USAGE STRONGLY ADVISED)

0:

To train our net on custom dataset some changes in different parts of the code are needed (like the minimum number of keypoints to display the image, the architecture of the net--> output size, ...)

To do this I created a config.py file in the cfg directory, it looks like this:

`import numpy as np`
`num_keys=4` --->number of keypoints (at least 3)
`BODY_PARTS_KPT_IDS = [[0,1],[0, 2], [0, 3]]`--->connections (at least 3)
`BODY_PARTS_PAF_IDS = ([0, 1], [2, 3],[4,5])`--->just continue the sequence when adding more connections
`min_score=0.05`-->min score to display results
`kpt_names = ['head', 'left_engine','right_engine','bottom_engine']`
`sigmas = np.array([0.5, 0.5, 0.5,0.5],dtype=np.float32) / 10.0`--->the length of sigmas has to be the number of keypoints--> use them for tracking persons id in video (between frames) 

`save=True`-->save the output image
`print_pose_entries=True`-->print pose_entries 
`print_allkeypoints=True`-->print all keypoints

1:

 Download pre-trained MobileNet v1 weights `mobilenet_sgd_68.848.pth.tar` from: <https://github.com/marvis/pytorch-mobilenet> (sgd option). If this doesn't work, download from [GoogleDrive](https://drive.google.com/file/d/18Ya27IAhILvBHqV_tDp0QjDFvsNNy-hv/view?usp=sharing) (these are the weights to start with). 

2:

Convert train annotations in internal format. Run `python scripts/prepare_train_labels.py --labels <COCO_HOME>/annotations/<your_json_file>.json`. It will produce `prepared_train_annotation.pkl` with converted in internal format annotations. 

3:

To train from MobileNet weights, run `python poseFINAL/pose-estimation/train.py --train-images-folder poseFINAL/trainimages/ --prepared-train-labels prepared_train_annotation.pkl --val-labels poseFINAL/annotations/documentation-test.json --val-images-folder poseFINAL/trainimages/ --checkpoint-path poseFINAL/mobilenet_sgd_68.848.pth.tar --from-mobilenet --batch-size 1 --checkpoint-after 100`  (do not care about the Warnings ).

--> ON DOCKER the training will stop after 400 iterations due to (i think) dataloader freeziong so you can use less workers, expand the shared memory or just run the below script manually for several times.

To train from your checkpoint:

`python poseFINAL/pose-estimation/train.py --train-images-folder poseFINAL/trainimages/ --prepared-train-labels prepared_train_annotation.pkl --val-labels poseFINAL/annotations/documentation-test.json --val-images-folder poseFINAL/trainimages/ --checkpoint-path default_checkpoints/checkpoint_iter_200.pth  --batch-size 1 --checkpoint-after 100 --weights-only`



For person pose estimation there are very good pretrained weights here <https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth> 

download them if you want with wget.



To do inference and test the weights run `python poseFINAL/pose-estimation/demo.py --checkpoint-path default_checkpoints/checkpoint_iter_200.pth  --images poseFINAL/trainimages/s1.jpg`



example of output:

```
[[0.       1.       2.       3.       5.583798 4.      ]]
[[ 53.          69.           0.81349993   0.        ]
 [ 76.          93.           0.67765409   1.        ]
 [ 80.          45.           0.81795776   2.        ]
 [148.          69.           0.72270733   3.        ]]
```

the first array is the pose_entries where there are all the object detected:

['head', 'left_engine','right_engine','bottom_engine']

[    0.                1.                  2.                        3.                    5.583798       4.      ]

the numbers are the indexes of the keypoint detected in allkeypoints array (-1 meand no keypoints found for that category in that object).

For example the first 0 means that the keypoint related to the head is allkeypoints[0] which is 

[ 53.          69.           0.81349993   0.        ]

this keypoint is in (x=53,y= 69) position with 0.81 score

--images <path>:

->in the config.py set save to True

it will save the output image as out.jpg

--video <path>:

->in the config.py set save to False

If you pass a webcam path it will stream the output othervise it saves the video

![out (4)](C:\Users\Utente\Desktop\POSE-ESTIMATION\doc-img\out (4).jpg)





# Face Yaw, Roll, Pitch, from face keypoints

**The repository where I took the opencv function to do this is here: <https://github.com/jerryhouuu/Face-Yaw-Roll-Pitch-from-Pose-Estimation-using-OpenCV> (the only .py file)**

**-->custom_pose folder**

-**facerotationoriginal.py** is the original file taken from the repository in case it will change.

-**rotation.py** is the modified file for the submarine--> it outputs an image rot.py taken as input the image from the path in landmark.txt where there are the coordinates of the keypoints.

how it works:

You need at least 4 keypoints coordinates in an image and the relative 3D positions of them in the space.

in the code:

the image_points array you have to put the coordinates (2D) of the keypoints.

in model_points array you have to put the relative 3D positions of them in the space:

    model_points = np.array([
                            (0.0, length, 0.0),             # head 3D coord.
                            (-(length*14)/53,(length*37)/53, 0.0),     # Left engine 3D coord.
                            ((length*14)/53, (length*37)/53, 0.0),      # Right engine 3D coord.
                            (0.0, 0.0, 0.0),    # bottom engine 3D coord.                       
                        ])
where length is the distance calcolated from the image between head and bottom (to adjust the distance factor) but I've seen that it works even if you set the length to a manually set one (for example 100).

In this case the bottom will be the axis origin.

Given this two  matrixes it computes through some math (and here I let functions from opencv library do the work) the euler angles.

(there is some math to understand and explore if someone wants)

![rot](C:\Users\Utente\Desktop\POSE-ESTIMATION\custom_pose\poseFINAL\rot.jpg)

**demo-anglesub.py** and  **demo-face.py** are like demo.py but with this angle output added to pose estimation. They put in the image_points matrix the detected points through pose estimation automatically and that they even display it live if you have a webcam access.

(-->ON JETSON NANO in one place of the code replace cv2.cv2.SOLVEPNP_ITERACTIVE with cv2.SOLVEPNP_ITERACTIVE and of course always adjust the paths)

![out](C:\Users\Utente\Desktop\POSE-ESTIMATION\doc-img\out.jpg)

(live webcam inference with Jetson nano)



### **Compile model with Tensorrt**-->much more faster

I used a library that helps you to convert your net  to onnx model and compile it with tensorrt (you have to install tensorrt too).

**The library is called torch2trt and this is the repository: <https://github.com/NVIDIA-AI-IOT/torch2trt>**

to install it: 

```
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
sudo python setup.py install
```

go to tensorrt folder in custom_pose and open **test-tensorrt.py** and see the comments, anyway:

to save the model:

-->`from torch2trt import torch2trt #import library`

-->`model_trt = torch2trt(net, [data]) #IT CREATES THE COMPILED VERSION OF YOUR MODEL, IT TAKES A WHILE`

-->`torch.save(model_trt.state_dict(), 'net_trt.pth') #TO SAVE THE WEIGHTS OF THE COMPILED MODEL WICH ARE DIFFERENT FROM THE PREVIOUS ONES`

to use the model:

-->`from torch2trt import TRTModule #import a class`

-->`model_trt = TRTModule() #the compiled model istance`

-->`model_trt.load_state_dict(torch.load('net_trt.pth')) #load the compiled weights in the compiled model`

