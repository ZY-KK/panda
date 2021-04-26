import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn import Linear, GRU, Conv2d, Dropout, MaxPool2d, BatchNorm1d, LSTM, RNN, Conv1d
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax

class Net(nn.Module):
    def __init__(self, channel_in, channel_out) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            
            nn.Conv2d(
                in_channels=channel_in,    
                out_channels=128,  
                kernel_size=5,    
                stride=1,         
                padding=2,       
            ),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   
        )
        self.conv2 = nn.Sequential(
            
            nn.Conv2d(
                in_channels=128,    
                out_channels=256,  
                kernel_size=5,    
                stride=1,         
                padding=2,       
            ),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   
        )
        self.conv3 = nn.Sequential(
            
            nn.Conv2d(
                in_channels=256,    
                out_channels=512,  
                kernel_size=5,    
                stride=1,         
                padding=2,       
            ),
            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)   
        )
        self.fc = nn.Sequential(   #full connection layers.
            nn.Linear(512,channel_out),

        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)           
        x = self.conv3(x)
        x = x.view(x.size(0),-1)
        output = self.output(x)     
        return output


