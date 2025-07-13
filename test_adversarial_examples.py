import os

import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import models
from models import MNIST_target_net
import torchvision.utils as vutils

use_cuda=True
image_nc=1
batch_size = 128

gen_input_nc = image_nc

if __name__ == '__main__':
    # 定义保存图片的文件夹
    images_dir = './images'
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # Define what device we are using
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # load the pretrained model
    pretrained_model = "./MNIST_target_model.pth"
    target_model = MNIST_target_net().to(device)
    target_model.load_state_dict(torch.load(pretrained_model))
    target_model.eval()

    # load the generator of adversarial examples
    pretrained_generator_path = './models/netG_epoch_60.pth'
    pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
    pretrained_neu_generator_path = './models/neu_netG_epoch_60.pth'
    pretrained_neu_G = models.Generator(gen_input_nc, image_nc).to(device)
    pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
    pretrained_neu_G.load_state_dict(torch.load(pretrained_neu_generator_path))
    pretrained_G.eval()
    pretrained_neu_G.eval()

    # test adversarial examples in MNIST training dataset
    mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    train_dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    num_correct_G = 0
    num_correct_neu = 0
    num_total = 0
    saved_images = 0
    for i, data in enumerate(train_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        # 生成干扰图像
        perturbation_G = pretrained_G(test_img)
        perturbation_neu = pretrained_neu_G(test_img)

        perturbation_G = torch.clamp(perturbation_G, -0.3, 0.3)
        perturbation_neu = torch.clamp(perturbation_neu, -0.3, 0.3)

        adv_img = perturbation_G + test_img
        adv_img_neu = perturbation_neu + test_img

        adv_img = torch.clamp(adv_img, 0, 1)
        adv_img_neu = torch.clamp(adv_img_neu, 0, 1)

        # 模型在干扰训练数据集的准确率
        pred_lab_G = torch.argmax(target_model(adv_img),1)
        pred_lab_neu = torch.argmax(target_model(adv_img_neu),1)

        num_correct_G += torch.sum(pred_lab_G==test_label,0)
        num_correct_neu += torch.sum(pred_lab_neu==test_label,0)

        num_total += test_label.size(0)

        # 保存图片
        if saved_images < 10:
            for img_idx in range(batch_size):
                if saved_images < 10:
                    original_img_path = f'./images/original_img_{saved_images}.png'
                    adv_img_path = f'./images/adv_img_G_{saved_images}.png'
                    adv_img_neu_path = f'./images/adv_img_neu_G_{saved_images}.png'

                    vutils.save_image(test_img[img_idx], original_img_path)
                    vutils.save_image(adv_img[img_idx], adv_img_path)
                    vutils.save_image(adv_img_neu[img_idx], adv_img_neu_path)

                    saved_images += 1
                else:
                    break

    print('netG:')
    print('num_correct: ', num_correct_G.item(), 'num_total: ', num_total)
    print('accuracy of adv imgs in training set: %f\n'%(num_correct_G.item()/len(mnist_dataset)))
    print('neu_netG:')
    print('num_correct: ', num_correct_neu.item(), 'num_total: ', num_total)
    print('accuracy of adv imgs in training set: %f\n'%(num_correct_neu.item()/len(mnist_dataset)))

    # test adversarial examples in MNIST testing dataset
    mnist_dataset_test = torchvision.datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor(), download=True)
    test_dataloader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=False, num_workers=1)
    num_correct_G = 0
    num_correct_neu = 0
    num_total = 0
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        # 生成干扰图像
        perturbation_G = pretrained_G(test_img)
        perturbation_neu = pretrained_neu_G(test_img)

        perturbation_G = torch.clamp(perturbation_G, -0.3, 0.3)
        perturbation_neu = torch.clamp(perturbation_neu, -0.3, 0.3)

        adv_img = perturbation_G + test_img
        adv_img_neu = perturbation_neu + test_img

        adv_img = torch.clamp(adv_img, 0, 1)
        adv_img_neu = torch.clamp(adv_img_neu, 0, 1)

        # 模型在干扰训练数据集的准确率
        pred_lab_G = torch.argmax(target_model(adv_img), 1)
        pred_lab_neu = torch.argmax(target_model(adv_img_neu), 1)

        num_correct_G += torch.sum(pred_lab_G == test_label, 0)
        num_correct_neu += torch.sum(pred_lab_neu == test_label, 0)

        num_total += test_label.size(0)

    print('netG:')
    print('num_correct: ', num_correct_G.item(), 'num_total: ', num_total)
    print('accuracy of adv imgs in testing set: %f\n'%(num_correct_G.item()/len(mnist_dataset)))
    print('neu_netG:')
    print('num_correct: ', num_correct_neu.item(), 'num_total: ', num_total)
    print('accuracy of adv imgs in testing set: %f\n'%(num_correct_neu.item()/len(mnist_dataset)))


    #
    # # load the generator of adversarial examples
    # pretrained_generator_path = './models/neu_netG_epoch_60.pth'
    # pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
    # pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
    # pretrained_G.eval()
    #
    # # test adversarial examples in MNIST training dataset
    # mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    # train_dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    # num_correct = 0
    # num_total = 0
    # for i, data in enumerate(train_dataloader, 0):
    #     test_img, test_label = data
    #     test_img, test_label = test_img.to(device), test_label.to(device)
    #     # 生成干扰图像
    #     perturbation = pretrained_G(test_img)
    #     perturbation = torch.clamp(perturbation, -0.3, 0.3)
    #     adv_img = perturbation + test_img
    #     adv_img = torch.clamp(adv_img, 0, 1)
    #     # 模型在干扰训练数据集的准确率
    #     pred_lab = torch.argmax(target_model(adv_img), 1)
    #     num_correct += torch.sum(pred_lab == test_label, 0)
    #     num_total += test_label.size(0)
    #
    # print('neu_netG:')
    # print('num_correct: ', num_correct.item(), 'num_total: ', num_total)
    # print('accuracy of adv imgs in training set: %f\n' % (num_correct.item() / len(mnist_dataset)))
    #
    # # test adversarial examples in MNIST testing dataset
    # mnist_dataset_test = torchvision.datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor(),
    #                                                 download=True)
    # test_dataloader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=False, num_workers=1)
    # num_correct = 0
    # num_total = 0
    # for i, data in enumerate(test_dataloader, 0):
    #     test_img, test_label = data
    #     test_img, test_label = test_img.to(device), test_label.to(device)
    #     perturbation = pretrained_G(test_img)
    #     perturbation = torch.clamp(perturbation, -0.3, 0.3)
    #     adv_img = perturbation + test_img
    #     adv_img = torch.clamp(adv_img, 0, 1)
    #     # 模型在干扰测试数据集的准确率
    #     pred_lab = torch.argmax(target_model(adv_img), 1)
    #     num_correct += torch.sum(pred_lab == test_label, 0)
    #     num_total += test_label.size(0)
    #
    # print('num_correct: ', num_correct.item(), 'num_total: ', num_total)
    # print('accuracy of adv imgs in testing set: %f\n' % (num_correct.item() / len(mnist_dataset_test)))

