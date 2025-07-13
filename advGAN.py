import torch.nn as nn
import torch
import numpy as np
import models
import torch.nn.functional as F
import torchvision
import os
from torchvision.utils import save_image
models_path = './models/'


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AdvGAN_Attack:
    def __init__(self,
                 device,
                 model,
                 model_num_labels,
                 image_nc,
                 box_min,
                 box_max):
        output_nc = image_nc
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model  # 分类器
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max
        self.eps = 1e-6
        self.threshold = 0.5

        self.gen_input_nc = image_nc
        self.netG = models.Generator(self.gen_input_nc, image_nc).to(device)
        self.netDisc = models.Discriminator(image_nc).to(device)

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=0.001)

        if not os.path.exists(models_path):
            os.makedirs(models_path)


    def train_batch(self, x, labels):
        # optimize D
        for i in range(1):
            perturbation = self.netG(x)

            # add a clipping trick
            adv_images = torch.clamp(perturbation, -0.3, 0.3) + x
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            self.optimizer_D.zero_grad()
            pred_real = self.netDisc(x)
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            loss_D_real.backward()

            pred_fake = self.netDisc(adv_images.detach())
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            loss_D_fake.backward()
            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()

        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()

            # cal G's loss in GAN
            pred_fake = self.netDisc(adv_images)
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            # calculate perturbation norm
            C = 0.1
            # 计算第二范数，约束扰动越小越好
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
            # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

            activations = []
            # 定义前向hook来收集激活（不使用detach）
            def get_activations_hook(activations_list):
                def hook(module, input, output):
                    if isinstance(module, nn.Conv2d):
                        pooled = F.adaptive_avg_pool2d(output, (1, 1)).squeeze(-1).squeeze(-1)
                        activations_list.append(pooled)
                    elif isinstance(module, nn.Linear):
                        activations_list.append(output)

                return hook
            hooks = []
            for name, layer in self.model.named_modules():
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    hook = layer.register_forward_hook(get_activations_hook(activations))
                    hooks.append(hook)

            # 前向传播
            logits_model = self.model(adv_images)

            for hook in hooks:
                hook.remove()

            # 计算神经损失
            neural_penalty = 0.0
            for activation in activations:
                batch_max, _ = activation.max(dim=0, keepdim=True)
                batch_min, _ = activation.min(dim=0, keepdim=True)
                normalized = (activation - batch_min) / (batch_max - batch_min + self.eps)
                penalty = torch.clamp(self.threshold - normalized, min=0)
                neural_penalty += penalty.mean()

            probs_model = F.softmax(logits_model, dim=1)
            # 创建单位矩阵，大小为10 * 10
            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]

            # C&W loss function
            real = torch.sum(onehot_labels * probs_model, dim=1)
            # torch.max返回最大值和索引
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other)
            # kappa = 0
            loss_adv = torch.max(real - other, zeros)  # 逐位比较
            loss_adv = torch.sum(loss_adv)
            '''
            one hot : 1 0 0 
            probs model: 0.1 0.2 0.7
            real: 0.1
            other: 0.7
            '''

            # maximize cross_entropy loss
            # loss_adv = -F.mse_loss(logits_model, onehot_labels)
            # loss_adv = - F.cross_entropy(logits_model, labels)

            adv_lambda = 10
            pert_lambda = 1
            beta = 10
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb + beta * neural_penalty
            loss_G.backward()
            self.optimizer_G.step()

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item(), neural_penalty.item()

    def train(self, train_dataloader, epochs):
        for epoch in range(1, epochs+1):

            if epoch == 50:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.0001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.0001)
            if epoch == 80:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.00001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.00001)
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            loss_neural_sum = 0

            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch, loss_neural_batch = \
                    self.train_batch(images, labels)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch
                loss_neural_sum += loss_neural_batch

            # print statistics
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %.3f, loss_adv: %.3f, loss_neural: %.3f\n" %
                  (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch, loss_neural_sum/num_batch))

            # save generator
            if epoch%20==0:
                netG_file_name = models_path + 'neu_netG_epoch_' + str(epoch) + '.pth'
                torch.save(self.netG.state_dict(), netG_file_name)


# def calculate_neural_loss_direct(perturbed_images, target_model, threshold=0.5, eps=1e-6):
#     """
#     直接计算神经损失，避免hook导致的梯度问题
#     """
#     activations = []
#
#     # 定义前向hook来收集激活（不使用detach）
#     def get_activation_hook(activations_list):
#         def hook(module, input, output):
#             if isinstance(module, nn.Conv2d):
#                 # 对卷积层进行全局平均池化
#                 pooled = F.adaptive_avg_pool2d(output, (1, 1)).squeeze(-1).squeeze(-1)
#                 activations_list.append(pooled)
#             elif isinstance(module, nn.Linear):
#                 activations_list.append(output)
#
#         return hook
#
#     # 注册临时hook
#     hooks = []
#     for name, layer in target_model.named_modules():
#         if isinstance(layer, (nn.Conv2d, nn.Linear)):
#             hook = layer.register_forward_hook(get_activation_hook(activations))
#             hooks.append(hook)
#
#     # 前向传播
#     _ = target_model(perturbed_images)
#
#     # 移除hook
#     for hook in hooks:
#         hook.remove()
#
#     # 计算神经损失
#     total_penalty = 0.0
#     for activation in activations:
#         # 归一化激活
#         batch_max, _ = activation.max(dim=0, keepdim=True)
#         batch_min, _ = activation.min(dim=0, keepdim=True)
#         normalized = (activation - batch_min) / (batch_max - batch_min + eps)
#
#         # 计算惩罚
#         penalty = torch.clamp(threshold - normalized, min=0)
#         total_penalty += penalty.mean()
#
#     return total_penalty