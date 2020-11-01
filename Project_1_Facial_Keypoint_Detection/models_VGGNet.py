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
        ## https://cs231n.github.io/convolutional-networks/
      
        ## output size 
        ### W : width
        ### F : Filter size 
        ### S : Stride
        #### (W-F)/S +1 
        #### (224-3)/1 +1 = 26
        self.conv1 = nn.Conv2d(in_channels = 1,  out_channels = 32,  kernel_size = 3)
        self.conv2 = nn.Conv2d(in_channels = 32,  out_channels = 32,  kernel_size = 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(in_channels = 32,  out_channels = 32,  kernel_size = 3)
        self.conv4 = nn.Conv2d(in_channels = 32,  out_channels = 32,  kernel_size = 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(in_channels = 32,  out_channels = 64,  kernel_size = 3)
        self.conv6 = nn.Conv2d(in_channels = 64,  out_channels = 64,  kernel_size = 3)
        self.conv7 = nn.Conv2d(in_channels = 64,  out_channels = 64,  kernel_size = 3)
        self.pool3 = nn.MaxPool2d(2, 2)  
        
        self.conv8 = nn.Conv2d(in_channels = 64,  out_channels = 128,  kernel_size = 3)
        self.conv9 = nn.Conv2d(in_channels = 128,  out_channels = 128,  kernel_size = 3)
        self.conv10 = nn.Conv2d(in_channels = 128,  out_channels = 128,  kernel_size = 3)
        self.pool4 = nn.MaxPool2d(2, 2)  
        

        
        
        #---------------#
        # Dense lauer 1 #
        #---------------# 
        # 20 outputs * the 5*5 filtered/pooled map size
        self.fc1 = nn.Linear( 128 * 8 * 8, 2048)
        self.fc1_drop = nn.Dropout(p=0.3)
        
        #---------------#
        # Dense layer 2 #
        #---------------#
        self.fc2 = nn.Linear(2048, 2048)
        self.fc2_drop = nn.Dropout(p=0.2)
        
        #----------------#
        # Output layer 3 #
        #----------------#
        self.fc3 = nn.Linear(2048, 136)
        
        
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
      
        # 4 conv/relu + pool layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.pool3(x)
        
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.pool4(x)
  
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = self.fc1_drop(F.relu(self.fc1(x)))
        
        x = self.fc2_drop(F.relu(self.fc2(x)))
        
        x = self.fc3(x)
        
        # final output
        return x
