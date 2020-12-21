import torch
import torch.autograd as autograd
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from torch.autograd import Variable

from util.image_pool import ImagePool


###############################################################################
# Functions
###############################################################################

class ContentLoss():
    def initialize(self, loss):
        self.criterion = loss

    def get_loss(self, fakeIm, realIm):
        return self.criterion(fakeIm, realIm)

    def __call__(self, fakeIm, realIm):
        return self.get_loss(fakeIm, realIm)


class PerceptualLoss():

    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        model = model.eval()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def initialize(self, loss):
        with torch.no_grad():
            self.criterion = loss
            self.contentFunc = self.contentFunc()
            self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_loss(self, fakeIm, realIm):
        fakeIm = (fakeIm + 1) / 2.0
        realIm = (realIm + 1) / 2.0
        fakeIm[0, :, :, :] = self.transform(fakeIm[0, :, :, :])
        realIm[0, :, :, :] = self.transform(realIm[0, :, :, :])
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return 0.006 * torch.mean(loss) + 0.5 * nn.MSELoss()(fakeIm, realIm)

    def __call__(self, fakeIm, realIm):
        return self.get_loss(fakeIm, realIm)


class GANLoss(nn.Module):
    def __init__(self, use_l1=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_l1:
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor.cuda()

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class DiscLoss(nn.Module):
    def name(self):
        return 'DiscLoss'

    def __init__(self):
        super(DiscLoss, self).__init__()

        self.criterionGAN = GANLoss(use_l1=False)
        self.fake_AB_pool = ImagePool(50)

    def get_g_loss(self, net, fakeB, realB):
        # First, G(A) should fake the discriminator
        pred_fake = net.forward(fakeB)
        return self.criterionGAN(pred_fake, 1)

    def get_loss(self, net, fakeB, realB):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero
        self.pred_fake = net.forward(fakeB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, 0)

        # Real
        self.pred_real = net.forward(realB)
        self.loss_D_real = self.criterionGAN(self.pred_real, 1)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def __call__(self, net, fakeB, realB):
        return self.get_loss(net, fakeB, realB)


class RelativisticDiscLoss(nn.Module):
    def name(self):
        return 'RelativisticDiscLoss'

    def __init__(self):
        super(RelativisticDiscLoss, self).__init__()

        self.criterionGAN = GANLoss(use_l1=False)
        self.fake_pool = ImagePool(50)  # create image buffer to store previously generated images
        self.real_pool = ImagePool(50)

    def get_g_loss(self, net, fakeB, realB):
        # First, G(A) should fake the discriminator
        self.pred_fake = net.forward(fakeB)

        # Real
        self.pred_real = net.forward(realB)
        errG = (self.criterionGAN(self.pred_real - torch.mean(self.fake_pool.query()), 0) +
                self.criterionGAN(self.pred_fake - torch.mean(self.real_pool.query()), 1)) / 2
        return errG

    def get_loss(self, net, fakeB, realB):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero
        self.fake_B = fakeB.detach()
        self.real_B = realB
        self.pred_fake = net.forward(fakeB.detach())
        self.fake_pool.add(self.pred_fake)

        # Real
        self.pred_real = net.forward(realB)
        self.real_pool.add(self.pred_real)

        # Combined loss
        self.loss_D = (self.criterionGAN(self.pred_real - torch.mean(self.fake_pool.query()), 1) +
                       self.criterionGAN(self.pred_fake - torch.mean(self.real_pool.query()), 0)) / 2
        return self.loss_D

    def __call__(self, net, fakeB, realB):
        return self.get_loss(net, fakeB, realB)


class RelativisticDiscLossLS(nn.Module):
    def name(self):
        return 'RelativisticDiscLossLS'

    def __init__(self):
        super(RelativisticDiscLossLS, self).__init__()

        self.criterionGAN = GANLoss(use_l1=True)
        self.fake_pool = ImagePool(50)  # create image buffer to store previously generated images
        self.real_pool = ImagePool(50)

    def get_g_loss(self, net, fakeB, realB):
        # First, G(A) should fake the discriminator
        self.pred_fake = net.forward(fakeB)

        # Real
        self.pred_real = net.forward(realB)
        errG = (torch.mean((self.pred_real - torch.mean(self.fake_pool.query()) + 1) ** 2) +
                torch.mean((self.pred_fake - torch.mean(self.real_pool.query()) - 1) ** 2)) / 2
        return errG

    def get_loss(self, net, fakeB, realB):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero
        self.fake_B = fakeB.detach()
        self.real_B = realB
        self.pred_fake = net.forward(fakeB.detach())
        self.fake_pool.add(self.pred_fake)

        # Real
        self.pred_real = net.forward(realB)
        self.real_pool.add(self.pred_real)

        # Combined loss
        self.loss_D = (torch.mean((self.pred_real - torch.mean(self.fake_pool.query()) - 1) ** 2) +
                       torch.mean((self.pred_fake - torch.mean(self.real_pool.query()) + 1) ** 2)) / 2
        return self.loss_D

    def __call__(self, net, fakeB, realB):
        return self.get_loss(net, fakeB, realB)


class DiscLossLS(DiscLoss):
    def name(self):
        return 'DiscLossLS'

    def __init__(self):
        super(DiscLossLS, self).__init__()
        self.criterionGAN = GANLoss(use_l1=True)

    def get_g_loss(self, net, fakeB, realB):
        return DiscLoss.get_g_loss(self, net, fakeB)

    def get_loss(self, net, fakeB, realB):
        return DiscLoss.get_loss(self, net, fakeB, realB)


class DiscLossWGANGP(DiscLossLS):
    def name(self):
        return 'DiscLossWGAN-GP'

    def __init__(self):
        super(DiscLossWGANGP, self).__init__()
        self.LAMBDA = 10

    def get_g_loss(self, net, fakeB, realB):
        # First, G(A) should fake the discriminator
        self.D_fake = net.forward(fakeB)
        return -self.D_fake.mean()

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD.forward(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty

    def get_loss(self, net, fakeB, realB):
        self.D_fake = net.forward(fakeB.detach())
        self.D_fake = self.D_fake.mean()

        # Real
        self.D_real = net.forward(realB)
        self.D_real = self.D_real.mean()
        # Combined loss
        self.loss_D = self.D_fake - self.D_real
        gradient_penalty = self.calc_gradient_penalty(net, realB.data, fakeB.data)
        return self.loss_D + gradient_penalty

class MixGradientLoss():

    def meanGradientError(self,fakeIm, realIm):
        filter_x = np.array([[-1, -2, -2], [0, 0, 0], [1, 2, 1]])
        filter_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        # filter_x = torch.FloatTensor([[-1, -2, -2], [0, 0, 0], [1, 2, 1]]).cuda().view(1, 1, 3, 3).repeat(3, 3, 1, 1)
        # filter_y = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).cuda().view(1, 1, 3, 3).repeat(3, 3, 1, 1)

        fakeIm = fakeIm.detach().cpu().numpy()
        realIm = realIm.detach().cpu().numpy()

        # print(fakeIm.size())
        # print(filter_x.size())
        # fake_grad_x = torch.square(nn.functional.conv2d(fakeIm,filter_x,padding=1))
        # fake_grad_y = torch.square(nn.functional.conv2d(fakeIm,filter_y,padding=1))

        # real_grad_x = torch.square(nn.functional.conv2d(realIm,filter_x,padding=1))
        # real_grad_y = torch.square(nn.functional.conv2d(realIm,filter_y,padding=1))
        # print(fake_grad_x.size())

        fake_grad_x = np.square(cv2.filter2D(fakeIm[0, :, :, :], -1, filter_x))
        fake_grad_y = np.square(cv2.filter2D(fakeIm[0, :, :, :], -1, filter_y))

        real_grad_x = np.square(cv2.filter2D(realIm[0, :, :, :], -1, filter_x))
        real_grad_y = np.square(cv2.filter2D(realIm[0, :, :, :], -1, filter_y))

        # output_gradients = torch.sqrt(torch.add(fake_grad_x, fake_grad_y))
        # target_gradients = torch.sqrt(torch.add(real_grad_x, real_grad_y))

        output_gradients = torch.Tensor(np.sqrt(np.add(fake_grad_x, fake_grad_y)))
        target_gradients = torch.Tensor(np.sqrt(np.add(real_grad_x, real_grad_y)))

        output_gradients = output_gradients.cuda()
        target_gradients = target_gradients.cuda()

        return output_gradients, target_gradients

    def initialize(self, loss):
        self.criterion = loss
        self.transform = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        # self.transform = transforms.Normalize(mean=[0,0,0], std=[1,1,1])

    def get_loss(self, fakeIm, realIm):
        fakeIm = (fakeIm + 1) / 2.0
        realIm = (realIm + 1) / 2.0
        fakeIm[0, :, :, :] = self.transform(fakeIm[0, :, :, :])
        realIm[0, :, :, :] = self.transform(realIm[0, :, :, :])

        output_gradients, target_gradients = self.meanGradientError(fakeIm, realIm)
        loss = self.criterion(fakeIm, realIm)

        # x = nn.MSELoss()(output_gradients, target_gradients)
        # print("MGE Loss: ",x)
        # y = torch.mean(loss)
        # print("L1 Loss: ",y)

        return 0.8 * torch.mean(loss) + 0.5 * nn.MSELoss()(output_gradients, target_gradients)

    def __call__(self, fakeIm, realIm):
        return self.get_loss(fakeIm, realIm)


def get_loss(model):
    if model['content_loss'] == 'perceptual':
        content_loss = PerceptualLoss()
        content_loss.initialize(nn.MSELoss())
    elif model['content_loss'] == 'l1':
        content_loss = ContentLoss()
        content_loss.initialize(nn.L1Loss())
    elif model['content_loss'] == 'gradient' :
        content_loss = MixGradientLoss()
        content_loss.initialize(nn.MSELoss())
    else:
        raise ValueError("ContentLoss [%s] not recognized." % model['content_loss'])

    if model['disc_loss'] == 'wgan-gp':
        disc_loss = DiscLossWGANGP()
    elif model['disc_loss'] == 'lsgan':
        disc_loss = DiscLossLS()
    elif model['disc_loss'] == 'gan':
        disc_loss = DiscLoss()
    elif model['disc_loss'] == 'ragan':
        disc_loss = RelativisticDiscLoss()
    elif model['disc_loss'] == 'ragan-ls':
        disc_loss = RelativisticDiscLossLS()
    else:
        raise ValueError("GAN Loss [%s] not recognized." % model['disc_loss'])
    return content_loss, disc_loss
