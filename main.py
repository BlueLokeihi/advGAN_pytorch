import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from advGAN import AdvGAN_Attack
from models import Resnet18_MNIST

use_cuda=True
image_nc=1
epochs = 20
batch_size = 128
BOX_MIN = 0
BOX_MAX = 1

if __name__ == '__main__':
    # Define what device we are using
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    pretrained_model = "./MNIST_target_model.pth"
    targeted_model = Resnet18_MNIST().model.to(device)
    targeted_model.load_state_dict(torch.load(pretrained_model))
    targeted_model.eval()
    model_num_labels = 10

    # MNIST train dataset and dataloader declaration
    # transform will normalize the pixel values between 0 and 1
    mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    advGAN = AdvGAN_Attack(device,
                              targeted_model,
                              model_num_labels,
                              image_nc,
                              BOX_MIN,
                              BOX_MAX)

    advGAN.train(dataloader, epochs)
