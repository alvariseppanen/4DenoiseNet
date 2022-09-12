# !/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import imp
from traceback import print_tb

import __init__ as booger
import torch
import torch.nn as nn
import torch.nn.functional as F



from torch.nn.parameter import Parameter
from torch.nn import init
import math
import time

class KNNConvBlock(nn.Module):

    def __init__(self, in_chans=3, stem_size=32, drop_rate=0.0):
        super(KNNConvBlock, self).__init__()
        self.drop_rate = drop_rate
        kernel_knn_size = 3
        self.search = 5
        self.pre_search = 7
        self.range_weight = Parameter(torch.Tensor(in_chans, stem_size, *(kernel_knn_size, kernel_knn_size) ))
        init.kaiming_uniform_(self.range_weight, a = math.sqrt(5))
        self.pre_range_weight = Parameter(torch.Tensor(in_chans, stem_size, *(kernel_knn_size, kernel_knn_size) ))
        init.kaiming_uniform_(self.pre_range_weight, a = math.sqrt(5))
        self.out_channels = stem_size
        self.knn_nr = kernel_knn_size ** 2
        self.act1 = nn.ReLU(inplace=True)

    def KNN_conv(self, inputs, pre_inputs):
        B, H, W = inputs.shape[0], inputs.shape[-2], inputs.shape[-1]
        search_dim = self.search ** 2
        pad = int((self.search - 1) / 2)
        pre_search_dim = self.pre_search ** 2
        pre_pad = int((self.pre_search - 1) / 2)

        proj_range = (inputs[:, 0:1, ...].clone())
        pre_proj_range = (pre_inputs[:, 0:1, ...].clone())
        
        unfold_proj_range = F.unfold(proj_range,
                            kernel_size=(self.search, self.search),
                            padding=(pad, pad))
        unfold_inputs = F.unfold(inputs,
                            kernel_size=(self.search, self.search),
                            padding=(pad, pad))
        unfold_pre_proj_range = F.unfold(pre_proj_range,
                            kernel_size=(self.pre_search, self.pre_search),
                            padding=(pre_pad, pre_pad))
        unfold_pre_inputs = F.unfold(pre_inputs,
                            kernel_size=(self.pre_search, self.pre_search),
                            padding=(pre_pad, pre_pad))

        center = int((search_dim - 1) / 2)
        difference = torch.sqrt((unfold_proj_range - unfold_proj_range[:, center:center + 1, ...]) ** 2)
        pre_difference = torch.sqrt((unfold_pre_proj_range - unfold_proj_range[:, center:center + 1, ...]) ** 2) 
        difference[:, center:center + 1, ...] = -1
        _, knn_idx = difference.topk(self.knn_nr, dim=1, largest=False)
        _, pre_knn_idx = pre_difference.topk(self.knn_nr, dim=1, largest=False)
        
        new_knn_idx = torch.cat((knn_idx,
                            knn_idx + search_dim,
                            knn_idx + 2 * search_dim,
                            knn_idx + 3 * search_dim,
                            knn_idx + 4 * search_dim), 1)

        pre_new_knn_idx = torch.cat((pre_knn_idx,
                            pre_knn_idx + pre_search_dim,
                            pre_knn_idx + 2 * pre_search_dim,
                            pre_knn_idx + 3 * pre_search_dim,
                            pre_knn_idx + 4 * pre_search_dim), 1)
        
        unfold_inputs = torch.gather(input=unfold_inputs, dim=1, index=new_knn_idx)
        unfold_pre_inputs = torch.gather(input=unfold_pre_inputs, dim=1, index=pre_new_knn_idx)

        anchors = torch.flatten(inputs, start_dim=2)[:, 1:4, None, :]
        x_points = unfold_pre_inputs[:, None, 1*self.knn_nr:2*self.knn_nr, :]
        y_points = unfold_pre_inputs[:, None, 2*self.knn_nr:3*self.knn_nr, :]
        z_points = unfold_pre_inputs[:, None, 3*self.knn_nr:4*self.knn_nr, :]
        xyz_points = torch.cat((x_points, y_points, z_points), 1)
        d = xyz_points - anchors

        # convert d vectors to spherical system
        x_points = d[:, 0, :, :].unsqueeze(dim=1)
        y_points = d[:, 1, :, :].unsqueeze(dim=1)
        z_points = d[:, 2, :, :].unsqueeze(dim=1)
        xy = x_points**2 + y_points**2
        d[:, 0, :, :] = torch.sqrt(xy + z_points**2)
        d[:, 1, :, :] = torch.arctan2(torch.sqrt(xy), z_points**2) # elevation angle defined from Z-axis down
        d[:, 2, :, :] = torch.arctan2(y_points**2, x_points**2)

        d = torch.flatten(d, start_dim=1, end_dim=2)
        unfold_pre_inputs = torch.cat((unfold_pre_inputs[:, :self.knn_nr, :], d, unfold_pre_inputs[:, -self.knn_nr:, :]), 1)
        
        output = torch.matmul(self.range_weight.view(self.out_channels, -1), unfold_inputs).view(B, self.out_channels, H, W)
        pre_output = torch.matmul(self.pre_range_weight.view(self.out_channels, -1), unfold_pre_inputs).view(B, self.out_channels, H, W)

        return output, pre_output

    def forward(self, x, pre_x):
        x, pre_x = self.KNN_conv(x, pre_x)
        x = self.act1(x)
        pre_x = self.act1(pre_x)

        return x, pre_x

class MGABlock(nn.Module):

    def __init__(self, in_size=64, out_size=64):
        super(MGABlock, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, 1, bias=True)
        self.conv2 = nn.Conv2d(in_size, 1, 1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def attention(self, x, pre_x):
        
        flow_feat_map = self.conv2(pre_x)
        flow_feat_map = nn.Sigmoid()(flow_feat_map)
        spatial_attentioned_feat = flow_feat_map * x

        feat_vec = self.avg_pool(spatial_attentioned_feat)
        feat_vec = self.conv1(feat_vec)
        feat_vec = nn.Softmax(dim=1)(feat_vec) * feat_vec.shape[1]
        channel_attentioned_feat = spatial_attentioned_feat * feat_vec

        final_feat = channel_attentioned_feat + x
        return final_feat

    def forward(self, x, pre_x):
        x = self.attention(x, pre_x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        resA = self.conv4(resA2)
        resA = self.act4(resA)
        resA3 = self.bn3(resA)

        concat = torch.cat((resA1,resA2,resA3),dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)
        resA = shortcut + resA


        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)

            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB

class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(in_filters//4 + 2*out_filters, out_filters, (3,3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (2,2), dilation=2,padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)


        self.conv4 = nn.Conv2d(out_filters*3,out_filters,kernel_size=(1,1))
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)

        upB = torch.cat((upA,skip),dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = self.conv2(upE1)
        upE = self.act2(upE)
        upE2 = self.bn2(upE)

        upE = self.conv3(upE2)
        upE = self.act3(upE)
        upE3 = self.bn3(upE)

        concat = torch.cat((upE1,upE2,upE3),dim=1)
        upE = self.conv4(concat)
        upE = self.act4(upE)
        upE = self.bn4(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE


class NN(nn.Module):
    def __init__(self, nclasses, params):
        super(NN, self).__init__()
        self.nclasses = nclasses

        self.input_size = 5
        
        self.KNNBlock = KNNConvBlock(self.input_size, 32)
        self.resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False)
        self.resBlock_pre = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False)
        self.resBlock7 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=False)
        self.upBlock6 = UpBlock(2 * 2 * 32, 32, 0.2, drop_out=False)
        self.attentionBlock = MGABlock(64, 64) # motion guided attention

        self.logits = nn.Conv2d(32, nclasses, kernel_size=(1, 1))

    def forward(self, x, pre_x):
        KNNBlock, pre_KNNBlock = self.KNNBlock(x, pre_x)

        down, down_skip = self.resBlock1(KNNBlock)
        pre_down, _ = self.resBlock_pre(pre_KNNBlock)
        down = self.attentionBlock(down, pre_down)
        bottom = self.resBlock7(down)
        up = self.upBlock6(bottom, down_skip)

        logits = self.logits(up)
        logits = F.softmax(logits, dim=1)
        return logits

