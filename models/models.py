import numpy as np
import torch.nn as nn
from skimage.measure import compare_ssim as SSIM

from util.metrics import PSNR


class DeblurModel(nn.Module):
    def __init__(self):
        super(DeblurModel, self).__init__()

    def get_input(self, data):
        amap_s1 = data['amap_s1']
        blurred_s1 = data['blurred_s1']
        sharp_s1 = data['sharp_s1']
        d_amap_s1 = data['d_amap_s1']
        amap_s2 = data['amap_s2']
        blurred_s2 = data['blurred_s2']
        sharp_s2 = data['sharp_s2']
        d_amap_s2 = data['d_amap_s2']
        amap_s3 = data['amap_s3']
        blurred_s3 = data['blurred_s3']
        sharp_s3 = data['sharp_s3']
        d_amap_s3 = data['d_amap_s3']
        amap_s1, blurred_s1, sharp_s1, d_amap_s1 = amap_s1.cuda(), blurred_s1.cuda(), sharp_s1.cuda(), d_amap_s1.cuda()
        amap_s2, blurred_s2, sharp_s2, d_amap_s2 = amap_s2.cuda(), blurred_s2.cuda(), sharp_s2.cuda(), d_amap_s2.cuda()
        amap_s3, blurred_s3, sharp_s3, d_amap_s3 = amap_s3.cuda(), blurred_s3.cuda(), sharp_s3.cuda(), d_amap_s3.cuda()
        return amap_s1, blurred_s1, sharp_s1, d_amap_s1, amap_s2, blurred_s2, sharp_s2, d_amap_s2, amap_s3, blurred_s3, sharp_s3, d_amap_s3

    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)

    def get_images_and_metrics(self, inp, output, target) -> (float, float, np.ndarray):
        inp = (inp + 1) * 0.5
        output = (output + 1) * 0.5
        tagret = (tagret + 1) * 0.5
        inp = self.tensor2im(inp)
        fake = self.tensor2im(output.data)
        real = self.tensor2im(target.data)
        psnr = PSNR(fake, real)
        ssim = SSIM(fake, real, multichannel=True)
        vis_img = np.hstack((inp, fake, real))
        return psnr, ssim, vis_img


def get_model(model_config):
    return DeblurModel()
