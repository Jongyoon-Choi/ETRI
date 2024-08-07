'''
AI Fashion Coordinator
(Baseline For Fashion-How Challenge)

MIT License

Copyright (C) 2022, Integrated Intelligence Research Section, ETRI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Update: 2024.04.20.
'''
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary

# ResNet(baseline)
class ResExtractor(nn.Module):
    """Feature extractor based on ResNet structure
        Selectable from resnet18 to resnet152

    Args:
        resnetnum: Desired resnet version
                    (choices=['18','34','50','101','152'])
        pretrained: 'True' if you want to use the pretrained weights provided by Pytorch,
                    'False' if you want to train from scratch.
    """

    def __init__(self, resnetnum='50', pretrained=True):
        super(ResExtractor, self).__init__()

        if resnetnum == '18':
            self.resnet = models.resnet18(pretrained=pretrained)
        elif resnetnum == '34':
            self.resnet = models.resnet34(pretrained=pretrained)
        elif resnetnum == '50':
            self.resnet = models.resnet50(pretrained=pretrained)
        elif resnetnum == '101':
            self.resnet = models.resnet101(pretrained=pretrained)
        elif resnetnum == '152':
            self.resnet = models.resnet152(pretrained=pretrained)

        self.modules_front = list(self.resnet.children())[:-2]
        self.model_front = nn.Sequential(*self.modules_front)
        
    def front(self, x):
        """ In the resnet structure, input 'x' passes through conv layers except for fc layers. """
        return self.model_front(x)


class Baseline_ResNet_emo(nn.Module):
    """ Classification network of emotion categories based on ResNet18 structure. """
    
    def __init__(self):
        super(Baseline_ResNet_emo, self).__init__()

        self.encoder = ResExtractor('18')
        self.avg_pool = nn.AvgPool2d(kernel_size=7)

        self.daily_linear = nn.Linear(512, 6)
        self.gender_linear = nn.Linear(512, 5)
        self.embel_linear = nn.Linear(512, 3)

    def forward(self, x):
        """ Forward propagation with input 'x' """
        feat = self.encoder.front(x['image'])
        flatten = self.avg_pool(feat).squeeze()

        out_daily = self.daily_linear(flatten)
        out_gender = self.gender_linear(flatten)
        out_embel = self.embel_linear(flatten)

        return out_daily, out_gender, out_embel

# MNet(baseline)
class MnExtractor(nn.Module):
    """Feature extractor based on MobileNetv2 structure
    Args:
        pretrained: 'True' if you want to use the pretrained weights provided by Pytorch,
                    'False' if you want to train from scratch.
    """

    def __init__(self, pretrained=True):
        super(MnExtractor, self).__init__()

        self.net = models.mobilenet_v2(pretrained=pretrained)
        self.modules_front = list(self.net.children())[:-1]
        self.model_front = nn.Sequential(*self.modules_front)

    def front(self, x):
        """ In the resnet structure, input 'x' passes through conv layers except for fc layers. """
        return self.model_front(x)


class Baseline_MNet_emo(nn.Module):
    """ Classification network of emotion categories based on MobileNetv2 structure. """
    
    def __init__(self):
        super(Baseline_MNet_emo, self).__init__()

        self.encoder = MnExtractor()
        self.avg_pool = nn.AvgPool2d(kernel_size=7)

        self.daily_linear = nn.Linear(1280, 6)
        self.gender_linear = nn.Linear(1280, 5)
        self.embel_linear = nn.Linear(1280, 3)

    def forward(self, x):
        """ Forward propagation with input 'x' """
        feat = self.encoder.front(x['image'])
        flatten = self.avg_pool(feat).squeeze()

        out_daily = self.daily_linear(flatten)
        out_gender = self.gender_linear(flatten)
        out_embel = self.embel_linear(flatten)

        return out_daily, out_gender, out_embel

# MNet v3
class Mnv3Extractor(nn.Module):
    """Feature extractor based on MobileNet v3 structure
    Args:
        pretrained: 'True' if you want to use the pretrained weights provided by Pytorch,
                    'False' if you want to train from scratch.
    """

    def __init__(self, pretrained=True):
        super(Mnv3Extractor, self).__init__()

        self.net = models.mobilenet_v3_large(pretrained=pretrained)
        self.modules_front = list(self.net.children())[:-2]
        self.model_front = nn.Sequential(*self.modules_front)

    def front(self, x):
        """ In the resnet structure, input 'x' passes through conv layers except for fc layers. """
        return self.model_front(x)


class MNetv3_emo(nn.Module):
    """ Classification network of emotion categories based on MobileNet v3 structure. """
    
    def __init__(self):
        super(MNetv3_emo, self).__init__()

        self.encoder = Mnv3Extractor()
        self.avg_pool = nn.AvgPool2d(kernel_size=7)

        self.daily_linear = nn.Linear(960, 6)
        self.gender_linear = nn.Linear(960, 5)
        self.embel_linear = nn.Linear(960, 3)

    def forward(self, x):
        """ Forward propagation with input 'x' """
        feat = self.encoder.front(x['image'])
        flatten = self.avg_pool(feat).squeeze()

        out_daily = self.daily_linear(flatten)
        out_gender = self.gender_linear(flatten)
        out_embel = self.embel_linear(flatten)

        return out_daily, out_gender, out_embel
    
# DenseNet
class DenseExtractor(nn.Module):
    """Feature extractor based on DenseNet structure
        Selectable from densenet121 to densenet201

    Args:
        densenetnum: Desired densenet version
                    (choices=['121','161','169','201'])
        pretrained: 'True' if you want to use the pretrained weights provided by Pytorch,
                    'False' if you want to train from scratch.
        memory_efficient: 'True' if you want to efficient densenet,
                    'False' if you want to original densenet.

    """

    def __init__(self, densenetnum='121', pretrained=True, memory_efficient=True):
        super(DenseExtractor, self).__init__()

        if densenetnum == '121':
            self.densenet = models.densenet121(pretrained=pretrained, memory_efficient=memory_efficient)
        elif densenetnum == '161':
            self.densenet = models.densenet161(pretrained=pretrained, memory_efficient=memory_efficient)
        elif densenetnum == '169':
            self.densenet = models.densenet169(pretrained=pretrained, memory_efficient=memory_efficient)
        elif densenetnum == '201':
            self.densenet = models.densenet201(pretrained=pretrained, memory_efficient=memory_efficient)

        self.modules_front = list(self.densenet.children())[:-1]
        self.model_front = nn.Sequential(*self.modules_front)

    def front(self, x):
        """ In the densenet structure, input 'x' passes through conv layers except for fc layers. """
        return self.model_front(x)
    

class DenseNet_emo(nn.Module):
    """ Classification network of emotion categories based on DenseNet121 structure. """
    
    def __init__(self):
        super(DenseNet_emo, self).__init__()

        self.encoder = DenseExtractor('121')
        self.avg_pool = nn.AvgPool2d(kernel_size=7)

        self.daily_linear = nn.Linear(1024, 6)
        self.gender_linear = nn.Linear(1024, 5)
        self.embel_linear = nn.Linear(1024, 3)

    def forward(self, x):
        """ Forward propagation with input 'x' """
        feat = self.encoder.front(x['image'])
        flatten = self.avg_pool(feat).squeeze()

        out_daily = self.daily_linear(flatten)
        out_gender = self.gender_linear(flatten)
        out_embel = self.embel_linear(flatten)

        return out_daily, out_gender, out_embel

# RegNet
class RegExtractor(nn.Module):
    """Feature extractor based on Regnet structure
    Args:
        pretrained: 'True' if you want to use the pretrained weights provided by Pytorch,
                    'False' if you want to train from scratch.
    """

    def __init__(self, pretrained=True):
        super(RegExtractor, self).__init__()

        self.net = models.regnet_y_800mf(pretrained=pretrained)
        self.modules_front = list(self.net.children())[:-2]
        self.model_front = nn.Sequential(*self.modules_front)

    def front(self, x):
        """ In the resnet structure, input 'x' passes through conv layers except for fc layers. """
        return self.model_front(x)


class RegNet_emo(nn.Module):
    """ Classification network of emotion categories based on Regnet structure. """
    
    def __init__(self):
        super(RegNet_emo, self).__init__()

        self.encoder = RegExtractor()
        self.avg_pool = nn.AvgPool2d(kernel_size=7)

        self.daily_linear = nn.Linear(784, 6)
        self.gender_linear = nn.Linear(784, 5)
        self.embel_linear = nn.Linear(784, 3)

    def forward(self, x):
        """ Forward propagation with input 'x' """
        feat = self.encoder.front(x['image'])
        flatten = self.avg_pool(feat).squeeze()

        out_daily = self.daily_linear(flatten)
        out_gender = self.gender_linear(flatten)
        out_embel = self.embel_linear(flatten)

        return out_daily, out_gender, out_embel


# EfficientNet, mnasnet, mobilenetv3, regnet 테스트 예정
# C:\Users\admin\anaconda3\Lib\site-packages\torchvision\models

if __name__ == '__main__':
    # # 모델 로드
    # model = models.squeezenet1_1(pretrained=True)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # # 모델 요약 출력
    # summary(model, (3, 224, 224))

    pass
