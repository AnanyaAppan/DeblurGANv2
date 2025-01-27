from models.encoder import Encoder
from models.bgdecoder import BGDecoder
from models.fgdecoder import FGDecoder
import torch
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, ConvTranspose2d, Sigmoid
from models.resnet_unit import ResidualBlock

class PDecoder(Module):   
    def __init__(self):
        super(PDecoder, self).__init__()

        self.layer1 = Sequential(
            Conv2d(128*3, 128, kernel_size = 1, stride = 1),
            ResidualBlock(128, 128, apply_activation=True),
            ResidualBlock(128, 128, apply_activation=True),
            ResidualBlock(128, 128, apply_activation=True),
            ConvTranspose2d(128, 64, kernel_size = 5, stride = 2, padding = 2, output_padding = 1, dilation = 1),
        )

        self.layer2 = Sequential(
            Conv2d(64*3, 64, kernel_size = 1, stride = 1),
            ResidualBlock(64, 64, apply_activation=True),
            ResidualBlock(64, 64, apply_activation=True),
            ResidualBlock(64, 64, apply_activation=True),
            ConvTranspose2d(64, 32, kernel_size = 5, stride = 2, padding = 2, output_padding = 1, dilation = 1),
        )


        self.layer3 = Sequential(
            Conv2d(32*2 + 32, 32, kernel_size = 1, stride = 1),
            ResidualBlock(32, 32, apply_activation=True),
            ResidualBlock(32, 32, apply_activation=True),
            ResidualBlock(32, 32, apply_activation=True),
            ConvTranspose2d(32, 1, kernel_size = 5, stride = 1, padding = 2, dilation = 1),
            Conv2d(1, 3, kernel_size=1, stride=1),
        )

        self.fgdecoder = FGDecoder()
        self.bgdecoder = BGDecoder()

        


    # Defining the forward pass    
    def forward(self, encoder_output1, encoder_output2, encoder_output3, fg_l1, fg_l2, fg_l3,bg_l1, bg_l2, bg_l3):
        x = torch.cat([encoder_output3, fg_l1, bg_l1], dim = 1)
        x = self.layer1(x)
        x += encoder_output2
        x = torch.cat([x, fg_l2, bg_l2], dim = 1)
        x = self.layer2(x)
        x += encoder_output1
        x = torch.cat([x, fg_l3, bg_l3], dim = 1)
        x = self.layer3(x)
        return x