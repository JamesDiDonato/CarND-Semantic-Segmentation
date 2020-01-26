# **Semantic Pixel Segmentation** 

### Completed for Udacity Self Driving Car Engineer - 2019/02

---

This project was completed in Feburary '19 as part of my enrollment in the Self Driving Car Engineer Nanodegree offered by Udacity.

[//]: # (Image References)
[image1]: ./pictures/architecture.png "architecture"
[image2]: ./pictures/2u1640.gif "outputs"
[image3]: ./pictures/loss.png "training_loss"
[image4]: ./pictures/sufficient_result.png "success_example"
[image5]: ./pictures/insufficient_result.png "insuccess_example"

## Requirements

The goal of this project is to implement a Fully Convolutional Neural Network (FCN) that is capable of labelling the 'open road' pixels from images taken by a front facing automobile camera. The implementation must be capable of labelling a reliable number of pixels in the test images accurately such that the output could be confidently fed to a path planner within a real-world self driving car application. The KITTI dataset will be used in the training and testing of the FCN, and is available for download  [here](http://www.cvlibs.net/datasets/kitti/eval_road.php).

Here is an example of a succesfull and un-succesfull result:

#### Successfull:

![alt text][image4]

#### UnSuccesfull:

![alt text][image5]

## Building the Neural Network

A chosen FCN must be effective at solving the semantic segmentation task of labelling each individual pixel. The FCN used in following paper was chosen as a suitable network for its promising results : [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1605.06211.pdf). The proposed network learns a mapping of pixel to pixel and can operate on arbirtary-sized inputs. The architecture chains together a [VGG-16](https://arxiv.org/pdf/1409.1556.pdf) encoder , a 1x1 convolution layer, and a transposed 'de-convolutional' network to implement the end to end learning. Additionally, the 3rd and 4th pooling layers of VGG-16 are fed-forward as 'skip connections' to be concatenated with the up-sampling layers. With the skip connections, the network is able predict finer details while retaining high level semantic information. Stated othersise by Shelhamer, Long, and Darrell , "Combining fine layers and coarse layers lets the model make local predictions that respect global structure." Here is an image of the architecture: 

![alt text][image1]


## Implementation

Unsuprisingly, the FCN is implemented using tensorflow/python with most of the relevant functions in `main.py`.  Useful helper functions are defined in `helper.py`. VGG-16 network weights are loaded from a pre-trained network in `def load_vgg(...)`, and the layers of the neural network are constructed in the `def layers(...)` function in `main.py` . 

## Neural Network Training

The neural network is trained in the `def train_nn(...)` function within `main.py` and prints out the training loss for each training batch. The hyper-parameters were selected as followed based on experimentation:

Hyper-parameter | Value 
--- | --- 
EPOCHS | 30
BATCH_SIZE | 5
KEEP_PROB | 0.6
LEARNING_RATE | 0.0005
L2_REG | 1e-3
STD_DEV | 0.01

Losses are plotted for each training batch. As you can see the loss decreases over time on average.

![alt text][image3]


## Results

Here are some sample output images illustrating the effectivness of my implementation:

![alt text][image2]
