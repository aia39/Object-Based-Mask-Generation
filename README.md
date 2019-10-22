# Object-Based-Mask-Generation
Object based mask generator is used to generate mask from a video from dataset of Office activity recognition.Mainly the task of this is to differentiate the video from irrelevant background object so that network can concentrate into important objects to classify better.
It was based on Inception Resnet V2 feature extractor which performs best among all of the existed benchmarked model on COCO dataset.
Pipeline of the work is 
![Pipeline of the work](Architecture.jpg)
![Getting the mask](Process.jpg)

This is implemented on Python 3 and TensorFlow.Here 13 objects from COCO dataset has been included for segmentation as these objects are important in office activity dataset.It's based on Instance segmentation and then assign specific color to same type of objects which is almost similar for detecting same type of class.

Some example of our work 
![Raw video frame](type.jpg)
![Masked frame](type11.jpg)


The repository includes:
* Source code of Mask R-CNN built on Inception Resnet V2 backbone.

## Getting Started
1. Clone this repository
2. Install dependencies
   ```bash
   conda install -c anaconda tensorflow-gpu
   conda install -c anaconda pillow
   conda install -c anaconda opencv
   conda install -c anaconda matplotlib
   ```
3. Unrar the research.rar file which contains 5 files 

3. Download pre-trained weights,classes names and related file from the [Tensorflow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).Download 'mask_rcnn_inception_resnet_v2_atrous_coco' from 'COCO-trained models' table.Put the .rar file in 'object detection' folder after step 2 is done and unrar it. 

4. Run the 'masking.py' from command window
 ```bash
   python masking.py
   ```
You can follow [Mask_RCNN_Dependencies Installation](https://github.com/tensorflow/models.git) to install every dependency required for this project. 



## Requirements
Python 3.4, TensorFlow 1.3,anaconda,opencv and other common packages.

## Related Works
* You can see related works and projects on which we worked from [VIP Cup 2019](https://signalprocessingsociety.org/get-involved/video-image-processing-cup) where we placed 2nd Runners Up.Our team 'BUET Synapticans',we are undergrad students of Bangladesh University of Engineering and University.

## Acknowledgement 
Tons of code was downloaded from theirs repo
https://github.com/tensorflow/models
    
