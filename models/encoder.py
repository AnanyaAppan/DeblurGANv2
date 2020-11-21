import torch
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, ConvTranspose2d, Sigmoid
from models.resnet_unit import ResidualBlock

class Encoder(Module):   
    def __init__(self):
        super(Encoder, self).__init__()

        self.block1 = Sequential (
            Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            ResidualBlock(32, 32, apply_activation=True),
            ResidualBlock(32, 32, apply_activation=True),
            ResidualBlock(32, 32, apply_activation=True),
        )

        self.block2 = Sequential (
            Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            ResidualBlock(64, 64, apply_activation=True),
            ResidualBlock(64, 64, apply_activation=True),
            ResidualBlock(64, 64, apply_activation=True),
        )

        self.block3 = Sequential (
            Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            ResidualBlock(128, 128, apply_activation=True),
            ResidualBlock(128, 128, apply_activation=True),
            ResidualBlock(128, 128, apply_activation=True),
        )

    # Defining the forward pass    
    def forward(self, x):
        block1_out = self.block1(x)
        block2_out = self.block2(block1_out)
        block3_out = self.block3(block2_out)
        return (block1_out,block2_out,block3_out)

# test
# img = torch.rand((1,6,16,16))
# encoder = Encoder()
# out = encoder(img)
# print(out)