## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## This last layer output contains 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
      
        ## output size 
        ### W : width
        ### F : Filter size 
        ### S : Stride
        #### (W-F)/S +1 
        #### (224-3)/1 +1 = 26
        self.conv1 = nn.Conv2d(in_channels = 1,  out_channels = 32,  kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64,  kernel_size= 5)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size= 5)
        #self.conv4 = nn.Conv2d(in_channels = 6114, out_channels = 128, kernel_size= 5)
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        

        
        #---------------#
        # Dense lauer 1 #
        #---------------# 
        # 20 outputs * the 5*5 filtered/pooled map size
        self.fc1 = nn.Linear(128 * 24 * 24, 512)
        
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)
        
        #---------------#
        # Dense layer 2 #
        #---------------#
        self.fc2 = nn.Linear(512, 256)
        
        # dropout with p=0.2
        self.fc2_drop = nn.Dropout(p=0.2)
        
        
        #----------------#
        # Output layer 3 #
        #----------------#
        self.fc3 = nn.Linear(256, 136)
        
        
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
      
        # 4 conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        #x = self.pool(F.relu(self.conv4(x)))

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)
        
        # final output
        return x

    
    
'''   
        
        # 1 input image channel (grayscale), 10 output channels/feature maps
        # 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (28-3)/1 +1 = 26
        # the output Tensor for one image, will have the dimensions: (10, 26, 26)
        # after one pool layer, this becomes (10, 13, 13)
        self.conv1 = nn.Conv2d(1, 10, 3)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # second conv layer: 10 inputs, 20 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (13-3)/1 +1 = 11
        # the output tensor will have dimensions: (20, 11, 11)
        # after another pool layer this becomes (20, 5, 5); 5.5 is rounded down
        self.conv2 = nn.Conv2d(10, 20, 3)
        
        # 20 outputs * the 5*5 filtered/pooled map size
        self.fc1 = nn.Linear(20*5*5, 50)
        
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)
        
        # finally, create 10 output channels (for the 10 classes)
        self.fc2 = nn.Linear(50, 10)

    # define the feedforward behavior
    def forward(self, x):
        # two conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        # final output
        return x
'''