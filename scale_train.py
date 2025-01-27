import logging
from functools import partial

import cv2
import torch
import torch.optim as optim
import tqdm
import yaml
import os
from joblib import cpu_count
from torch.utils.data import DataLoader
from torch.autograd import Variable

from adversarial_trainer import GANFactory
from dataset import PairedDataset
from metric_counter import MetricCounter
from models.losses import get_loss
from models.models import get_model
from models.networks import get_nets
from schedulers import LinearDecay, WarmRestart
import time

cv2.setNumThreads(8)

class Trainer:

    def loss_with_attention(self, output, target, attention_map):
        loss = torch.mean((torch.mul(output,attention_map) - torch.mul(target,attention_map))**2)
        return loss

    def calculate_fg_loss(self, fg_decoder_output1, fg_decoder_output2, fg_decoder_output3, sharp_s1, sharp_s2, sharp_s3, amap_s1, amap_s2, amap_s3):
        loss = self.loss_with_attention(fg_decoder_output1, sharp_s1, amap_s1) + self.loss_with_attention(fg_decoder_output2, sharp_s2, amap_s2) + self.loss_with_attention(fg_decoder_output3, sharp_s3, amap_s3)
        return loss

    def calculate_bg_loss(self, bg_decoder_output1, bg_decoder_output2, bg_decoder_output3, sharp_s1, sharp_s2, sharp_s3, amap_s1, amap_s2, amap_s3):
        loss = self.loss_with_attention(bg_decoder_output1, sharp_s1, 1 - amap_s1) + self.loss_with_attention(bg_decoder_output2, sharp_s2, 1 - amap_s2) + self.loss_with_attention(bg_decoder_output3, sharp_s3, 1 - amap_s3)
        return loss

    def calculate_pri_loss(self, p_decoder_output1, p_decoder_output2, p_decoder_output3, sharp_s1, sharp_s2, sharp_s3):
        loss = self.criterionG(p_decoder_output1,sharp_s1) + self.criterionG(p_decoder_output2,sharp_s2) + self.criterionG(p_decoder_output3,sharp_s3)
        return loss

    def __init__(self, config, train: DataLoader, val: DataLoader):
        self.config = config
        self.train_dataset = train
        self.val_dataset = val
        self.adv_lambda = config['model']['adv_lambda']
        self.metric_counter = MetricCounter(config['experiment_desc'])
        self.warmup_epochs = config['warmup_num']
        self.input_time = 0
        self.output_time = 0
        self.loss_time = 0
        self.grad_descent_time = 0

    def train(self):
        self._init_params()
        for epoch in range(0, config['num_epochs']):
            if (epoch == self.warmup_epochs) and not (self.warmup_epochs == 0):
                self.netG.module.unfreeze()
                self.optimizer_G = self._get_optim(self.netG.parameters())
                self.scheduler_G = self._get_scheduler(self.optimizer_G)
            self._run_epoch(epoch)
            self._validate(epoch)
            self.scheduler_G.step()
            self.scheduler_D.step()

            if self.metric_counter.update_best_model():
                torch.save({
                    'model': self.netG.state_dict()
                }, 'best_{}.h5'.format(self.config['experiment_desc']))
            torch.save({
                'model': self.netG.state_dict()
            }, 'last_{}.h5'.format(self.config['experiment_desc']))
            print(self.metric_counter.loss_message())
            logging.debug("Experiment Name: %s, Epoch: %d, Loss: %s" % (
                self.config['experiment_desc'], epoch, self.metric_counter.loss_message()))
            print("time taken for getting input = %f" % (self.input_time))
            print("time taken for getting output = %f" % (self.output_time))
            print("time taken for getting loss = %f" % (self.loss_time))
            print("time taken for gradient descent = %f" % (self.grad_descent_time))

    def _run_epoch(self, epoch):
        self.metric_counter.clear()
        for param_group in self.optimizer_G.param_groups:
            lr = param_group['lr']

        epoch_size = config.get('train_batches_per_epoch') or len(self.train_dataset)
        tq = tqdm.tqdm(self.train_dataset, total=epoch_size)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        i = 0
        for data in tq:
            input_start = time.time()
            amap_s1, blurred_s1, sharp_s1, d_amap_s1, amap_s2, blurred_s2, sharp_s2, d_amap_s2, amap_s3, blurred_s3, sharp_s3, d_amap_s3 = self.model.get_input(data)
            input_end = time.time()
            self.input_time += input_end - input_start
            output_start = time.time()
            p_decoder_output1, fg_decoder_output1, bg_decoder_output1, p_decoder_output2, fg_decoder_output2, bg_decoder_output2,p_decoder_output3, fg_decoder_output3, bg_decoder_output3 = self.netG(blurred_s1, blurred_s2, blurred_s3, d_amap_s1, d_amap_s2, d_amap_s3)
            output_end = time.time()
            self.output_time += output_end - output_start
            # outputs = self.netG(inputs, attention_maps, downsampled_attention_maps)
            loss_start = time.time()
            fg_loss = self.calculate_fg_loss(fg_decoder_output1, fg_decoder_output2, fg_decoder_output3, sharp_s1, sharp_s2, sharp_s3, amap_s1, amap_s2, amap_s3)
            bg_loss = self.calculate_bg_loss(bg_decoder_output1, bg_decoder_output2, bg_decoder_output3, sharp_s1, sharp_s2, sharp_s3, amap_s1, amap_s2, amap_s3)
            loss_D = self._update_d(p_decoder_output3, sharp_s3)
            self.optimizer_G.zero_grad()
            # loss_content = self.criterionG(outputs, targets)
            pri_loss = self.calculate_pri_loss(p_decoder_output1, p_decoder_output2, p_decoder_output3, sharp_s1, sharp_s2, sharp_s3)
            loss_adv = self.adv_trainer.loss_g(p_decoder_output3, sharp_s3)
            loss_G = pri_loss + self.adv_lambda * loss_adv
            loss = fg_loss + bg_loss + loss_G
            loss_end = time.time()
            self.loss_time += loss_end - loss_start
            grad_desc_start = time.time()
            # fg_loss.backward(retain_graph=True)
            # bg_loss.backward(retain_graph=True)
            # loss_G.backward()
            loss.backward()
            self.optimizer_G.step()
            grad_desc_end = time.time()
            self.grad_descent_time += grad_desc_end - grad_desc_start
            self.metric_counter.add_losses(loss_G.item(), pri_loss.item(), loss_D)
            curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(blurred_s3, p_decoder_output3, sharp_s3)
            self.metric_counter.add_metrics(curr_psnr, curr_ssim)
            tq.set_postfix(loss=self.metric_counter.loss_message())
            if not i:
                self.metric_counter.add_image(img_for_vis, tag='train')
            i += 1
            if i > epoch_size:
                break
        tq.close()
        self.metric_counter.write_to_tensorboard(epoch)

    def _validate(self, epoch):
        self.metric_counter.clear()
        epoch_size = config.get('val_batches_per_epoch') or len(self.val_dataset)
        tq = tqdm.tqdm(self.val_dataset, total=epoch_size)
        tq.set_description('Validation')
        i = 0
        for data in tq:
            amap_s1, blurred_s1, sharp_s1, d_amap_s1, amap_s2, blurred_s2, sharp_s2, d_amap_s2, amap_s3, blurred_s3, sharp_s3, d_amap_s3 = self.model.get_input(data)
            p_decoder_output1, _, _, p_decoder_output2, _, _,p_decoder_output3, _, _ = self.netG(blurred_s1, blurred_s2, blurred_s3, d_amap_s1, d_amap_s2, d_amap_s3)
            # outputs = self.netG(inputs, attention_maps, downsampled_attention_maps)
            # loss_content = self.criterionG(outputs, targets)
            pri_loss = self.calculate_pri_loss(p_decoder_output1, p_decoder_output2, p_decoder_output3, sharp_s1, sharp_s2, sharp_s3)
            loss_adv = self.adv_trainer.loss_g(p_decoder_output3, sharp_s3)
            loss_G = pri_loss + self.adv_lambda * loss_adv
            self.metric_counter.add_losses(loss_G.item(), pri_loss.item())
            curr_psnr, curr_ssim, img_for_vis = self.model.get_images_and_metrics(blurred_s3, p_decoder_output3, sharp_s3)
            self.metric_counter.add_metrics(curr_psnr, curr_ssim)
            if not i:
                self.metric_counter.add_image(img_for_vis, tag='val')
            i += 1
            if i > epoch_size:
                break
        tq.close()
        self.metric_counter.write_to_tensorboard(epoch, validation=True)

    def _update_d(self, outputs, targets):
        if self.config['model']['d_name'] == 'no_gan':
            return 0
        self.optimizer_D.zero_grad()
        loss_D = self.adv_lambda * self.adv_trainer.loss_d(outputs, targets)
        loss_D.backward(retain_graph=True)
        self.optimizer_D.step()
        return loss_D.item()

    def _get_optim(self, params):
        if self.config['optimizer']['name'] == 'adam':
            optimizer = optim.Adam(params, lr=self.config['optimizer']['lr'])
        elif self.config['optimizer']['name'] == 'sgd':
            optimizer = optim.SGD(params, lr=self.config['optimizer']['lr'])
        elif self.config['optimizer']['name'] == 'adadelta':
            optimizer = optim.Adadelta(params, lr=self.config['optimizer']['lr'])
        else:
            raise ValueError("Optimizer [%s] not recognized." % self.config['optimizer']['name'])
        return optimizer

    def _get_scheduler(self, optimizer):
        if self.config['scheduler']['name'] == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode='min',
                                                             patience=self.config['scheduler']['patience'],
                                                             factor=self.config['scheduler']['factor'],
                                                             min_lr=self.config['scheduler']['min_lr'])
        elif self.config['optimizer']['name'] == 'sgdr':
            scheduler = WarmRestart(optimizer)
        elif self.config['scheduler']['name'] == 'linear':
            scheduler = LinearDecay(optimizer,
                                    min_lr=self.config['scheduler']['min_lr'],
                                    num_epochs=self.config['num_epochs'],
                                    start_epoch=self.config['scheduler']['start_epoch'])
        else:
            raise ValueError("Scheduler [%s] not recognized." % self.config['scheduler']['name'])
        return scheduler

    @staticmethod
    def _get_adversarial_trainer(d_name, net_d, criterion_d):
        if d_name == 'no_gan':
            return GANFactory.create_model('NoGAN')
        elif d_name == 'patch_gan' or d_name == 'multi_scale':
            return GANFactory.create_model('SingleGAN', net_d, criterion_d)
        elif d_name == 'double_gan':
            return GANFactory.create_model('DoubleGAN', net_d, criterion_d)
        else:
            raise ValueError("Discriminator Network [%s] not recognized." % d_name)

    def _init_params(self):
        self.criterionG, criterionD = get_loss(self.config['model'])
        self.netG, netD = get_nets(self.config['model'])
        self.netG.cuda()
        self.netG.load_state_dict(torch.load("../weights/skip_scale/best_fpn.h5")['model'])
        self.adv_trainer = self._get_adversarial_trainer(self.config['model']['d_name'], netD, criterionD)
        self.model = get_model(self.config['model'])
        self.optimizer_G = self._get_optim(filter(lambda p: p.requires_grad, self.netG.parameters()))
        self.optimizer_D = self._get_optim(self.adv_trainer.get_params())
        self.scheduler_G = self._get_scheduler(self.optimizer_G)
        self.scheduler_D = self._get_scheduler(self.optimizer_D)

if __name__ == '__main__':
    with open('config/config.yaml', 'r') as f:
        config = yaml.load(f)

    batch_size = config.pop('batch_size')
    dataloader_start = time.time()
    get_dataloader = partial(DataLoader, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True)

    datasets = map(config.pop, ('train', 'val'))
    datasets = map(PairedDataset.from_config, datasets)
    train, val = map(get_dataloader, datasets)
    dataloader_end = time.time()
    data_loader_prep_time = dataloader_end - dataloader_start
    print("Time taken for preparing dataset = %f"%(data_loader_prep_time))
    trainer = Trainer(config, train=train, val=val)
    trainer.train()
