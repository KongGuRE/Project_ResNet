# ---------------------------------------------------------------------------- #
# An implementation of https://arxiv.org/pdf/1512.03385.pdf                    #
# See section 4.2 for the model architecture on CIFAR-10                       #
# Some part of the code was referenced from below                              #
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py   #
# ---------------------------------------------------------------------------- #
import datetime
import glob
import logging
import os
import random
from logging.config import dictConfig
from typing import Type, Union, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(message)s',
        }
    },
    'handlers': {
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'filename': 'debug.log',
            'formatter': 'default',
        },
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['file']
    }
})


def label_files(files):
    labels = []
    for file in files:
        if '\\NG\\' in file:
            labels.append(0)
        else:
            labels.append(1)
    return labels


class CustomDataset(Dataset):
    def __init__(self, files, labels, augmentation, valid):
        self.files = files
        self.labels = labels
        self.aug = augmentation
        self.valid = valid
        self.data = []

        for i in range(len(self.files)):
            sample = {'img': self.files[i], 'label': labels[i]}
            self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]['img']
        y = self.data[idx]['label']

        x = Image.open(x)
        x1 = self.aug(x)

        return {"img": np.array(x1, dtype='float32'), "labels": y}


# 3x3 convolution
# def conv3x3(in_channels, out_channels, stride=1):
#     return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#
#
# # Residual block
# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = conv3x3(in_channels, out_channels, stride)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(out_channels, out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.downsample = downsample
#
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         if self.downsample:
#             residual = self.downsample(x)
#         out += residual
#         out = self.relu(out)
#         return out
#
#
# # ResNet
# class ResNet(nn.Module):
#     def __init__(self, block, layers, num_classes=2):
#         super(ResNet, self).__init__()
#         self.in_channels = 64
#         self.conv = conv3x3(3, 64)
#         self.bn = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.layer1 = self.make_layer(block, 64, layers[0])
#         self.layer2 = self.make_layer(block, 128, layers[1], 2)
#         self.layer3 = self.make_layer(block, 256, layers[2], 2)
#         self.layer4 = self.make_layer(block, 512, layers[3], 2)
#         self.avg_pool = nn.AvgPool2d(32)
#         self.fc = nn.Linear(512, num_classes)
#
#     def make_layer(self, block, out_channels, blocks, stride=1):
#         downsample = None
#         if (stride != 1) or (self.in_channels != out_channels):
#             downsample = nn.Sequential(conv3x3(self.in_channels, out_channels, stride=stride),
#                                        nn.BatchNorm2d(out_channels))
#         layers = []
#         layers.append(block(self.in_channels, out_channels, stride, downsample))
#         self.in_channels = out_channels
#         for i in range(1, blocks):
#             layers.append(block(out_channels, out_channels))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.conv(x)
#         out = self.bn(out)
#         out = self.relu(out)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.avg_pool(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out


# ResNet18 BasicBlock class
class BasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super(BasicBlock, self).__init__()

        # 3x3 conv
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)  # 배치 정규화(batch normalization)
        self.relu = nn.ReLU(inplace=True)

        # 3x3 conv stride=1, padding=1
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)  # 배치 정규화(batch normalization)

        self.shortcut = nn.Sequential()  # identity인 경우
        if stride != 1:  # if stride is not 1, if not Identity mapping
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.conv2(self.bn2(out))
        out += self.shortcut(identity)  # skip connection
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock]],
            number_of_block: List[int],
            num_classes: int = 2
    ) -> None:
        super(ResNet, self).__init__()
        self.in_planes = 64

        # 64개의 3x3 필터(filter)를 사용
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, number_of_block[0], stride=1)
        self.layer2 = self._make_layer(block, 128, number_of_block[1], stride=2)
        self.layer3 = self._make_layer(block, 256, number_of_block[2], stride=2)
        self.layer4 = self._make_layer(block, 512, number_of_block[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block: Type[Union[BasicBlock]], planes: int, number_of_block: int, stride: int):
        strides = [stride] + [1] * (number_of_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes  # 다음 레이어를 위해 채널 수 변경
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.float()
        features = F.relu(self.bn1(self.conv1(x)))
        # out = self.maxpool(out)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        # print(features.shape)
        out = F.avg_pool2d(features, 28)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, features


# ResNet18
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def fifo_numpy_1D(fifo_numpy_data, numpy_append_data):
    fifo_numpy_data = np.delete(fifo_numpy_data, 0)
    fifo_numpy_data = np.append(fifo_numpy_data, numpy_append_data)
    return fifo_numpy_data


if __name__ == '__main__':

    _ck = True
    _img_ck = False

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print(os.getcwd())

    path = 'C:/DataSET/ImageData/P-TCP/P-TCP'
    os.listdir(path)

    train_files = path + '/Train/'
    test_files = path + '/Test/'

    train_files = glob.glob(train_files + '/*/*.jpg')
    test_files = glob.glob(test_files + '/*/*.jpg')

    # print(test_files[5])
    # temp = Image.open(test_files[5])
    # temp.show()

    train_labels = label_files(train_files)
    test_labels = label_files(test_files)

    # Image preprocessing modules

    torch.manual_seed(43)
    torch.cuda.manual_seed(43)
    random.seed(43)
    np.random.seed(43)

    # Hyper-parameters
    num_epochs = 100
    learning_rate = 0.001
    batch_size = 1024
    test_batch_size = 1
    img_size = 64

    # train_transforms = transforms.Compose([
    #     transforms.Resize(img_size),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.RandomAffine(45),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    # test_transforms = transforms.Compose([
    #     transforms.Resize(img_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])

    train_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    # CustomDataset
    train_dataset = CustomDataset(train_files, train_labels, augmentation=train_transforms, valid=False)
    val_dataset = CustomDataset(test_files, test_labels, augmentation=test_transforms, valid=True)
    # Data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, drop_last=True, num_workers=4)

    # print(train_labels)
    # for i, data in enumerate(train_loader):
    #     images, labels = data['img'].to(device), data['labels'].to(device)
    #     print (labels)

    if _ck:
        # for i, data in enumerate(train_loader):
        #     print(i)
        #     images, labels = data['img'].to(device), data['labels'].to(device)
        #     print("images.shape: " + str(images.shape))
        #     print("labels.shape: " + str(labels.shape))

        print('######### Dataset class created #########')
        print('Number of images: ', len(train_files) + len(test_files))
        print('train size in no of batch: ', len(train_loader))
        print("test size in no of batch: ", len(test_loader))
        print('train size: ', len(train_loader) * batch_size)
        print("test size: ", len(test_loader) * test_batch_size)

    if _img_ck:
        x = next(iter(train_loader))
        im = x['img'][0].permute(1, 2, 0)
        print(im.shape)
        plt.title(x['labels'][0])
        plt.imshow(im)
        plt.show()

    model = models.resnet18(pretrained=True)
    # model = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)
    model.fc = nn.Linear(512, 2)
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    test_acc = 0
    _ck_test_model: str = ""

    _acc_max = 0
    _acc_max_count = 0
    _loss_min = 1000000
    _loss_min_count = 0

    total = 0
    correct = 0

    while 1:

        print('\n######### Train the model #########\n')
        # Train the model
        model.train()
        total_step = len(train_loader)
        curr_lr = learning_rate
        correct = 0
        total = 0

        for epoch in tqdm(range(num_epochs)):

            loss_train_epoch = 0

            for i, data in enumerate(train_loader):
                images, labels = data['img'].to(device), data['labels'].to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)

                # Backward and optimize

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                loss_train_epoch += loss.item()
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            acc_train_epoch = 100. * correct / total
            print('\nTotal benign train accuracy: ', acc_train_epoch)
            print('Total benign train loss: ', loss_train_epoch)
            logging.debug("*epoch" + "|" + str(epoch) + "|" + 'Total benign train accuracy: ' + "|" + str(
                acc_train_epoch) + "|" + 'Total benign train loss: ' + "|" + str(loss_train_epoch) + "|")

            # Save the model checkpoint
            if _acc_max < acc_train_epoch:
                _acc_max = acc_train_epoch
                torch.save(model.state_dict(), r'.\ckpt\resnet' + str(epoch) + '.ckpt')
                logging.debug("Max accuracy .ckpt: " + str(epoch) + '.ckpt')

            elif acc_train_epoch == 100:
                _acc_max_count += 1
                _acc_max = acc_train_epoch
                torch.save(model.state_dict(), r'.\ckpt\resnet' + str(epoch) + '.ckpt')
                logging.debug("Max accuracy .ckpt: " + str(epoch) + '.ckpt')
                if _acc_max_count > 5:
                    if loss_train_epoch < 0.2:
                        _ck_test_model = "Max accuracy .ckpt: " + str(epoch) + '.ckpt'
                        break
            else:
                _acc_max_count = 0

            # Decay learning rate
            if _loss_min > loss_train_epoch:
                _loss_min = loss_train_epoch
                _loss_min_count = 0

            else:
                print("\nloss_min: ", _loss_min)
                print("loss_min_count: ", _loss_min_count)
                _loss_min_count += 1
                if acc_train_epoch > 90:
                    if _loss_min_count >= 20:
                        _loss_min_count = -5
                        curr_lr /= 3
                        update_lr(optimizer, curr_lr)
                        print("\nchange lr = ", curr_lr)

                # if epoch < 20:
                #     if _loss_min_count >= 10:
                #         _loss_min_count = -5
                #         curr_lr /= 3
                #         update_lr(optimizer, curr_lr)
                #         print("\nchange lr = ", curr_lr)
                #
                # elif epoch > 20:
                #     if _loss_min_count >= 5:
                #         _loss_min_count = -5
                #         curr_lr /= 3
                #         update_lr(optimizer, curr_lr)
                #         print("\nchange lr = ", curr_lr)

        # Test the model
        model.eval()
        total_result = 0
        with torch.no_grad():
            correct = 0
            total = 0
            Dj_time_c = 1
            for data in test_loader:
                images, labels = data['img'].to(device), data['labels'].to(device)
                start = datetime.datetime.now()
                outputs = model(images)
                end = datetime.datetime.now()
                result = end - start
                Dj_time_c = Dj_time_c + 1

                _, predicted = torch.max(outputs.data, 1)
                total_result = total_result + result.microseconds / 1000000
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            test_acc = 100 * correct / total

            print("===================================================")
            print('one time : ', result, end='\n')
            print('one time microseconds : ', result.microseconds, end='\n')
            print("total time : ", total_result)
            print('AVG time : ', total_result / Dj_time_c)
            print('Accuracy of the model on the test images: {} %'.format(test_acc))
            print("===================================================")

            logging.debug("**Test result -- " + "test_model |" + _ck_test_model + "|" +
                          "one time |" + str(result) + "|" +
                          "one time microseconds |" + str(result.microseconds) + "|" +
                          "total time |" + str(total_result) + "|" +
                          "AVG time |" + str(total_result / Dj_time_c) + "|" +
                          "'Accuracy of the model on the test images |" + str(test_acc) + "|")

        _loss_min_count = 30
        _acc_max_count = -10

        if test_acc == 100:
            print("Test Acc 100% epoch")
            break
