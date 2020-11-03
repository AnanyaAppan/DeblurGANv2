from torch.nn import Sequential, Module, ConvTranspose2d


class CNN(Module):   
    def __init__(self):
        super(CNN, self).__init__()

        self.layers = Sequential(
            # Conv2d(6, 32, kernel_size=5, stride=1, padding=2),
            ConvTranspose2d(3, 3, kernel_size=5, stride=1, padding=2), 
        )


    # Defining the forward pass    
    # def forward(self, img, prev_img):
    def forward(self,img, attention_map, downsampled_attention_map):
        x = self.layers(img)

        return x


