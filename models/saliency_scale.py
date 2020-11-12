from models.encoder import Encoder
from models.bgdecoder import BGDecoder
from models.fgdecoder import FGDecoder
from models.pdecoder import PDecoder
from models.human_aware import HumanAware
import torch
import torchvision.transforms as transforms
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, ConvTranspose2d, Sigmoid


class Saliency_Scale(Module):   

    def unfreeze(self):
        pass

    def _init_(self):
        super(Saliency_Scale, self)._init_()

        self.downsample = Sequential(
            Conv2d(3, 3, kernel_size=5, stride=4, padding=2),
        )

        self.upsample = Sequential(
            ConvTranspose2d(1,3,kernel_size = 5, stride = 2, padding = 2, output_padding = 1, dilation = 1)
        )


        self.layer = Sequential(
            Conv2d(3*2, 3, kernel_size = 1, stride = 1)
        )

        self.encoder = Encoder()
        self.fgdecoder = FGDecoder()
        self.bgdecoder = BGDecoder()
        self.pdecoder = PDecoder()
        self.human_aware = HumanAware()

    # Defining the forward pass    
    # def forward(self, img, prev_img):
    def forward(self,img1, img2, img3, downsampled_attention_map1, downsampled_attention_map2, downsampled_attention_map3):
        # First Image 
        attention_map_fg1 = downsampled_attention_map1
        attention_map_bg1 = 1 - attention_map_fg1
        encoder_input1 = img1
        primary_branch_input1 = self.encoder(encoder_input1)
        stacked_fg_attention1 = attention_map_fg1.clone()
        stacked_bg_attention1 = attention_map_bg1.clone()
        for i in range(7) :
            stacked_fg_attention1 = torch.cat((stacked_fg_attention1,stacked_fg_attention1),1)
            stacked_bg_attention1 = torch.cat((stacked_bg_attention1,stacked_bg_attention1),1)
        fg_branch_input1 = torch.mul(primary_branch_input1,stacked_fg_attention1)
        bg_branch_input1 = torch.mul(primary_branch_input1,stacked_bg_attention1)
        fg_decoder_output1, fg_l11, fg_l21, fg_l31 = self.fgdecoder(fg_branch_input1)
        bg_decoder_output1, bg_l11, bg_l21, bg_l31 = self.bgdecoder(bg_branch_input1)
        p_decoder_output1 = self.pdecoder(primary_branch_input, fg_l11, fg_l21, fg_l31, bg_l11, bg_l21, bg_l31)

        # Second Image
        upsampled_image1 = self.upsample(p_decoder_output1)
        input2 = torch.cat((upsampled_image1,img2),1)
        input2 = self.layer(input2)
        img2 = self.layer(input2)
        attention_map_fg2 = downsampled_attention_map2
        attention_map_bg2 = 1 - attention_map_fg2
        encoder_input2 = img2
        primary_branch_input2 = self.encoder(encoder_input2)
        stacked_fg_attention2 = attention_map_fg2.clone()
        stacked_bg_attention2 = attention_map_bg2.clone()
        for i in range(7) :
            stacked_fg_attention2 = torch.cat((stacked_fg_attention2,stacked_fg_attention2),1)
            stacked_bg_attention2 = torch.cat((stacked_bg_attention2,stacked_bg_attention2),1)
        fg_branch_input2 = torch.mul(primary_branch_input2,stacked_fg_attention2)
        bg_branch_input2 = torch.mul(primary_branch_input2,stacked_bg_attention2)
        fg_decoder_output2, fg_l12, fg_l22, fg_l32 = self.fgdecoder(fg_branch_input2)
        bg_decoder_output2, bg_l12, bg_l22, bg_l32 = self.bgdecoder(bg_branch_input2)
        p_decoder_output2 = self.pdecoder(primary_branch_input, fg_l12, fg_l22, fg_l32, bg_l12, bg_l22, bg_l32)

        # Third Image
        upsampled_image2 = self.upsample(p_decoder_output2)
        input3 = torch.cat((upsampled_image2,img3),1)
        input3 = self.layer(input3)
        img3 = self.layer(input3)
        attention_map_fg3 = downsampled_attention_map3
        attention_map_bg3 = 1 - attention_map_fg3
        encoder_input3 = img3
        primary_branch_input3 = self.encoder(encoder_input3)
        stacked_fg_attention3 = attention_map_fg3.clone()
        stacked_bg_attention3 = attention_map_bg3.clone()
        for i in range(7) :
            stacked_fg_attention3 = torch.cat((stacked_fg_attention3,stacked_fg_attention3),1)
            stacked_bg_attention3 = torch.cat((stacked_bg_attention3,stacked_bg_attention3),1)
        fg_branch_input3 = torch.mul(primary_branch_input3,stacked_fg_attention3)
        bg_branch_input3 = torch.mul(primary_branch_input3,stacked_bg_attention3)
        fg_decoder_output3, fg_l13, fg_l23, fg_l33 = self.fgdecoder(fg_branch_input3)
        bg_decoder_output3, bg_l13, bg_l23, bg_l33 = self.bgdecoder(bg_branch_input3)
        p_decoder_output3 = self.pdecoder(primary_branch_input, fg_l13, fg_l23, fg_l33, bg_l13, bg_l23, bg_l33)

        return p_decoder_output1, fg_decoder_output1, bg_decoder_output1, p_decoder_output2, fg_decoder_output2, bg_decoder_output2,p_decoder_output3, fg_decoder_output3, bg_decoder_output3