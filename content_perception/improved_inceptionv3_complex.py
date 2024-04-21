from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
import time
import math
import shutil
import os
import copy
import argparse
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
import csv
import pymysql


class Inception3(nn.Module):

    def __init__(self, num_classes=40, aux_logits=True, transform_input=False):
        super(Inception3, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionAplus(192, pool_features=32)
        self.Mixed_5c = InceptionAplus(256, pool_features=64)
        self.Mixed_5d = InceptionAplus(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionCplus(768, channels_7x7=128)
        self.Mixed_6c = InceptionCplus(768, channels_7x7=160)
        self.Mixed_6d = InceptionCplus(768, channels_7x7=160)
        self.Mixed_6e = InceptionCplus(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.fc = nn.Linear(2048, num_classes)

        # Initializes the weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.transform_input:  # 1
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, p=0.3,training=self.training)

        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)

        if self.training and self.aux_logits:
            return x, aux
        return x


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class InceptionAplus(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionAplus, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch1x3_1 = BasicConv2d(64, 64, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x1_1 = BasicConv2d(64, 64, kernel_size=(3, 1), padding=(1, 0))
        self.branch1x3_2 = BasicConv2d(64, 64, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x1_2 = BasicConv2d(64, 64, kernel_size=(3, 1), padding=(1, 0))

        #self.branch3x3_2 = BasicConv2d(48, 64, kernel_size=3, padding=1)
        #self.branch3x3_3 = BasicConv2d(64, 64, kernel_size=3, padding=1)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(48, 48, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(48, 48, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3_3a = BasicConv2d(48, 48, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_3b = BasicConv2d(48, 48, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        residual1 = branch3x3
        branch3x3 = self.branch1x3_1(branch3x3)
        branch3x3 = self.branch3x1_1(branch3x3)
        branch3x3 = branch3x3+residual1
        branch3x3 = self.relu(branch3x3)
        residual2 = branch3x3
        branch3x3 = self.branch1x3_2(branch3x3)
        branch3x3 = self.branch3x1_2(branch3x3)
        branch3x3 = branch3x3 + residual2
        branch3x3 = self.relu(branch3x3)

        branch3x3dbl_1 = self.branch3x3dbl_1(x)
        branch3x3dbl_2 = self.branch3x3dbl_1(x)
        branch3x3dbl_1=self.branch3x3_2a(branch3x3dbl_1)
        branch3x3dbl_1=self.branch3x3_2b(branch3x3dbl_1)
        branch3x3dbl_2 = self.branch3x3_3a(branch3x3dbl_2)
        branch3x3dbl_2 = self.branch3x3_3b(branch3x3dbl_2)

        branch3x3dbl = [
            branch3x3dbl_1,
            branch3x3dbl_2
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionCplus(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionCplus, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_4 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_5 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        residual1 = branch7x7dbl
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = branch7x7dbl + residual1
        residual2 = branch7x7dbl
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch7x7dbl = branch7x7dbl + residual2
        branch7x7dbl = self.relu(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


def inception_v3(pretrained=False, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        model = Inception3(**kwargs)
        # model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
        return model

    return Inception3(**kwargs)



def main():
    file_name = 'categories_places365.txt'
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)
    model = inception_v3()
    weights = torch.load('./checkpointimprovedinceptionv3.pth.tar_best.pth.tar', map_location=lambda storage, loc: storage)
    model.load_state_dict(weights["state_dict"])

    transform_test = transforms.Compose(
        [transforms.Resize([299, 299]),
         transforms.ToTensor(),
         transforms.Normalize((0.4576, 0.4411, 0.4080), (0.2689, 0.2669, 0.2849))
         ])

    testset = torchvision.datasets.ImageFolder(root=r'.\experiment1',
                                               transform=transform_test)
    # test_dataset = TestDataset(path_real, transform_test)
    # print(testset)
    idlist=[]
    with open('datacsvdatongrawfinal.csv', 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            id = row[0]
            idlist.append(id)
        f.close()


    model.eval()

    conn = pymysql.connect(host='127.0.0.1',
                           port=3306,
                           user='root',
                           password='241033',
                           db='datongresult', charset='utf8')
    cursor = conn.cursor()

    torch.tensor([1, 2])
    idnum=0
    for data, img_np in testset:
        img = torch.unsqueeze(data, 0)
        output = model(img)
        _, indices = torch.sort(output, descending=True)
        indices = indices.numpy()
        datals = []

        for idx in indices[0][:5]:
            datals.append(classes[idx])
        classfy1 = ""
        classfy2 = ""
        classfy3 = ""
        classfy4 = ""
        classfy5 = ""
        if datals[0]=="airport_terminal":
            classfy1="2"
        if datals[0]=="alley":
            classfy1="3"
        if datals[0]=="amusement_park":
            classfy1="3"
        if datals[0]=="apartment_building/outdoor":
            classfy1="2"
        if datals[0]=="aqueduct":
            classfy1="2"
        if datals[0]=="archaelogical_excavation":
            classfy1="1"
        if datals[0]=="art_gallery":
            classfy1="3"
        if datals[0] == "auto_factory":
            classfy1 = "2"
        if datals[0] == "badlands":
            classfy1 = "0"
        if datals[0] == "bakery/shop":
            classfy1 = "3"
        if datals[0] == "banquet_hall":
            classfy1 = "3"
        if datals[0] == "bazaar/outdoor":
            classfy1 = "3"
        if datals[0] == "boardwalk":
            classfy1 = "2"
        if datals[0] == "botanical_garden":
            classfy1 = "0"
        if datals[0] == "bridge":
            classfy1 = "2"
        if datals[0] == "building_facade":
            classfy1 = "2"
        if datals[0] == "burial_chamber":
            classfy1 = "1"
        if datals[0] == "butchers_shop":
            classfy1 = "3"
        if datals[0] == "butte":
            classfy1 = "0"
        if datals[0] == "canyon":
            classfy1 = "0"
        if datals[0] == "catacomb":
            classfy1 = "1"
        if datals[0] == "construction_site":
            classfy1 = "2"
        if datals[0] == "crosswalk":
            classfy1 = "2"
        if datals[0] == "delicatessen":
            classfy1 = "3"
        if datals[0] == "dining_hall":
            classfy1 = "3"
        if datals[0] == "downtown":
            classfy1 = "2"
        if datals[0] == "field/wild":
            classfy1 = "0"
        if datals[0] == "grotto":
            classfy1 = "1"
        if datals[0] == "mountain":
            classfy1 = "0"
        if datals[0] == "museum/indoor":
            classfy1 = "1"
        if datals[0] == "pagoda":
            classfy1 = "1"
        if datals[0] == "palace":
            classfy1 = "1"
        if datals[0] == "plaza":
            classfy1 = "2"
        if datals[0] == "river":
            classfy1 = "0"
        if datals[0] == "ruin":
            classfy1 = "1"
        if datals[0] == "sky":
            classfy1 = "0"
        if datals[0] == "street":
            classfy1 = "2"
        if datals[0] == "temple/asia":
            classfy1 = "1"
        if datals[0] == "throne_room":
            classfy1 = "1"
        if datals[0] == "valley":
            classfy1 = "0"

        if datals[1] == "airport_terminal":
            classfy2 = "2"
        if datals[1] == "alley":
            classfy2 = "3"
        if datals[1] == "amusement_park":
            classfy2 = "3"
        if datals[1] == "apartment_building/outdoor":
            classfy2 = "2"
        if datals[1] == "aqueduct":
            classfy2 = "2"
        if datals[1] == "archaelogical_excavation":
            classfy2 = "1"
        if datals[1] == "art_gallery":
            classfy2 = "3"
        if datals[1] == "auto_factory":
            classfy2 = "2"
        if datals[1] == "badlands":
            classfy2 = "0"
        if datals[1] == "bakery/shop":
            classfy2 = "3"
        if datals[1] == "banquet_hall":
            classfy2 = "3"
        if datals[1] == "bazaar/outdoor":
            classfy2 = "3"
        if datals[1] == "boardwalk":
            classfy2 = "2"
        if datals[1] == "botanical_garden":
            classfy2 = "0"
        if datals[1] == "bridge":
            classfy2 = "2"
        if datals[1] == "building_facade":
            classfy2 = "2"
        if datals[1] == "burial_chamber":
            classfy2 = "1"
        if datals[1] == "butchers_shop":
            classfy2 = "3"
        if datals[1] == "butte":
            classfy2 = "0"
        if datals[1] == "canyon":
            classfy2 = "0"
        if datals[1] == "catacomb":
            classfy2 = "1"
        if datals[1] == "construction_site":
            classfy2 = "2"
        if datals[1] == "crosswalk":
            classfy2 = "2"
        if datals[1] == "delicatessen":
            classfy2 = "3"
        if datals[1] == "dining_hall":
            classfy2 = "3"
        if datals[1] == "downtown":
            classfy2 = "2"
        if datals[1] == "field/wild":
            classfy2 = "0"
        if datals[1] == "grotto":
            classfy2 = "1"
        if datals[1] == "mountain":
            classfy2 = "0"
        if datals[1] == "museum/indoor":
            classfy2 = "1"
        if datals[1] == "pagoda":
            classfy2 = "1"
        if datals[1] == "palace":
            classfy2 = "1"
        if datals[1] == "plaza":
            classfy2 = "2"
        if datals[1] == "river":
            classfy2 = "0"
        if datals[1] == "ruin":
            classfy2 = "1"
        if datals[1] == "sky":
            classfy2 = "0"
        if datals[1] == "street":
            classfy2 = "2"
        if datals[1] == "temple/asia":
            classfy2 = "1"
        if datals[1] == "throne_room":
            classfy2 = "1"
        if datals[1] == "valley":
            classfy2 = "0"

        if datals[2] == "airport_terminal":
            classfy3 = "2"
        if datals[2] == "alley":
            classfy3 = "3"
        if datals[2] == "amusement_park":
            classfy3 = "3"
        if datals[2] == "apartment_building/outdoor":
            classfy3 = "2"
        if datals[2] == "aqueduct":
            classfy3 = "2"
        if datals[2] == "archaelogical_excavation":
            classfy3 = "1"
        if datals[2] == "art_gallery":
            classfy3 = "3"
        if datals[2] == "auto_factory":
            classfy3 = "2"
        if datals[2] == "badlands":
            classfy3 = "0"
        if datals[2] == "bakery/shop":
            classfy3 = "3"
        if datals[2] == "banquet_hall":
            classfy3 = "3"
        if datals[2] == "bazaar/outdoor":
            classfy3 = "3"
        if datals[2] == "boardwalk":
            classfy3 = "2"
        if datals[2] == "botanical_garden":
            classfy3 = "0"
        if datals[2] == "bridge":
            classfy3 = "2"
        if datals[2] == "building_facade":
            classfy3 = "2"
        if datals[2] == "burial_chamber":
            classfy3 = "1"
        if datals[2] == "butchers_shop":
            classfy3 = "3"
        if datals[2] == "butte":
            classfy3 = "0"
        if datals[2] == "canyon":
            classfy3 = "0"
        if datals[2] == "catacomb":
            classfy3 = "1"
        if datals[2] == "construction_site":
            classfy3 = "2"
        if datals[2] == "crosswalk":
            classfy3 = "2"
        if datals[2] == "delicatessen":
            classfy3 = "3"
        if datals[2] == "dining_hall":
            classfy3 = "3"
        if datals[2] == "downtown":
            classfy3 = "2"
        if datals[2] == "field/wild":
            classfy3 = "0"
        if datals[2] == "grotto":
            classfy3 = "1"
        if datals[2] == "mountain":
            classfy3 = "0"
        if datals[2] == "museum/indoor":
            classfy3 = "1"
        if datals[2] == "pagoda":
            classfy3 = "1"
        if datals[2] == "palace":
            classfy3 = "1"
        if datals[2] == "plaza":
            classfy3 = "2"
        if datals[2] == "river":
            classfy3 = "0"
        if datals[2] == "ruin":
            classfy3 = "1"
        if datals[2] == "sky":
            classfy3 = "0"
        if datals[2] == "street":
            classfy3 = "2"
        if datals[2] == "temple/asia":
            classfy3 = "1"
        if datals[2] == "throne_room":
            classfy3 = "1"
        if datals[2] == "valley":
            classfy3 = "0"

        if datals[3] == "airport_terminal":
            classfy4 = "2"
        if datals[3] == "alley":
            classfy4 = "3"
        if datals[3] == "amusement_park":
            classfy4 = "3"
        if datals[3] == "apartment_building/outdoor":
            classfy4 = "2"
        if datals[3] == "aqueduct":
            classfy4 = "2"
        if datals[3] == "archaelogical_excavation":
            classfy4 = "1"
        if datals[3] == "art_gallery":
            classfy4 = "3"
        if datals[3] == "auto_factory":
            classfy4 = "2"
        if datals[3] == "badlands":
            classfy4 = "0"
        if datals[3] == "bakery/shop":
            classfy4 = "3"
        if datals[3] == "banquet_hall":
            classfy4 = "3"
        if datals[3] == "bazaar/outdoor":
            classfy4 = "3"
        if datals[3] == "boardwalk":
            classfy4 = "2"
        if datals[3] == "botanical_garden":
            classfy4 = "0"
        if datals[3] == "bridge":
            classfy4 = "2"
        if datals[3] == "building_facade":
            classfy4 = "2"
        if datals[3] == "burial_chamber":
            classfy4 = "1"
        if datals[3] == "butchers_shop":
            classfy4 = "3"
        if datals[3] == "butte":
            classfy4 = "0"
        if datals[3] == "canyon":
            classfy4 = "0"
        if datals[3] == "catacomb":
            classfy4 = "1"
        if datals[3] == "construction_site":
            classfy4 = "2"
        if datals[3] == "crosswalk":
            classfy4 = "2"
        if datals[3] == "delicatessen":
            classfy4 = "3"
        if datals[3] == "dining_hall":
            classfy4 = "3"
        if datals[3] == "downtown":
            classfy4 = "2"
        if datals[3] == "field/wild":
            classfy4 = "0"
        if datals[3] == "grotto":
            classfy4 = "1"
        if datals[3] == "mountain":
            classfy4 = "0"
        if datals[3] == "museum/indoor":
            classfy4 = "1"
        if datals[3] == "pagoda":
            classfy4 = "1"
        if datals[3] == "palace":
            classfy4 = "1"
        if datals[3] == "plaza":
            classfy4 = "2"
        if datals[3] == "river":
            classfy4 = "0"
        if datals[3] == "ruin":
            classfy4 = "1"
        if datals[3] == "sky":
            classfy4 = "0"
        if datals[3] == "street":
            classfy4 = "2"
        if datals[3] == "temple/asia":
            classfy4 = "1"
        if datals[3] == "throne_room":
            classfy4 = "1"
        if datals[3] == "valley":
            classfy4 = "0"

        if datals[4] == "airport_terminal":
            classfy5 = "2"
        if datals[4] == "alley":
            classfy5 = "3"
        if datals[4] == "amusement_park":
            classfy5 = "3"
        if datals[4] == "apartment_building/outdoor":
            classfy5 = "2"
        if datals[4] == "aqueduct":
            classfy5 = "2"
        if datals[4] == "archaelogical_excavation":
            classfy5 = "1"
        if datals[4] == "art_gallery":
            classfy5 = "3"
        if datals[4] == "auto_factory":
            classfy5 = "2"
        if datals[4] == "badlands":
            classfy5 = "0"
        if datals[4] == "bakery/shop":
            classfy5 = "3"
        if datals[4] == "banquet_hall":
            classfy5 = "3"
        if datals[4] == "bazaar/outdoor":
            classfy5 = "3"
        if datals[4] == "boardwalk":
            classfy5 = "2"
        if datals[4] == "botanical_garden":
            classfy5 = "0"
        if datals[4] == "bridge":
            classfy5 = "2"
        if datals[4] == "building_facade":
            classfy5 = "2"
        if datals[4] == "burial_chamber":
            classfy5 = "1"
        if datals[4] == "butchers_shop":
            classfy5 = "3"
        if datals[4] == "butte":
            classfy5 = "0"
        if datals[4] == "canyon":
            classfy5 = "0"
        if datals[4] == "catacomb":
            classfy5 = "1"
        if datals[4] == "construction_site":
            classfy5 = "2"
        if datals[4] == "crosswalk":
            classfy5 = "2"
        if datals[4] == "delicatessen":
            classfy5 = "3"
        if datals[4] == "dining_hall":
            classfy5 = "3"
        if datals[4] == "downtown":
            classfy5 = "2"
        if datals[4] == "field/wild":
            classfy5 = "0"
        if datals[4] == "grotto":
            classfy5 = "1"
        if datals[4] == "mountain":
            classfy5 = "0"
        if datals[4] == "museum/indoor":
            classfy5 = "1"
        if datals[4] == "pagoda":
            classfy5 = "1"
        if datals[4] == "palace":
            classfy5 = "1"
        if datals[4] == "plaza":
            classfy5 = "2"
        if datals[4] == "river":
            classfy5 = "0"
        if datals[4] == "ruin":
            classfy5 = "1"
        if datals[4] == "sky":
            classfy5 = "0"
        if datals[4] == "street":
            classfy5 = "2"
        if datals[4] == "temple/asia":
            classfy5 = "1"
        if datals[4] == "throne_room":
            classfy5 = "1"
        if datals[4] == "valley":
            classfy5 = "0"

        sql = "update datong_ugc set classfy5 = %s where id = %s"
        cursor.execute(sql, (classfy5, idlist[idnum]))
        conn.commit()
        idnum=idnum+1

    conn.close()

if __name__ == '__main__':
    main()