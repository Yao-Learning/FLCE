import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision import models

import numpy as np
from sklearn.decomposition import PCA

from utils import weights_init

import fmodule


def get_basic_net(net, n_classes, input_size=None, input_channel=None):
    if net == "MLPNet":
        model = MLPNet(input_size, input_channel, n_classes)
    elif net == "LeNet":
        model = LeNet(input_size, input_channel, n_classes)
    elif net == "TFCNN":
        model = TFCNN(n_classes)
    elif net == "VGG8":
        model = VGG(8, n_classes, False)
    elif net == "VGG11":
        model = VGG(11, n_classes, False)
    elif net == "VGG13":
        model = VGG(13, n_classes, False)
    elif net == "VGG16":
        model = VGG(16, n_classes, False)
    elif net == "VGG19":
        model = VGG(19, n_classes, False)
    elif net == "VGG8-BN":
        model = VGG(8, n_classes, True)
    elif net == "VGG11-BN":
        model = VGG(11, n_classes, True)
    elif net == "VGG13-BN":
        model = VGG(13, n_classes, True)
    elif net == "VGG16-BN":
        model = VGG(16, n_classes, True)
    elif net == "VGG19-BN":
        model = VGG(19, n_classes, True)
    elif net == "ResNet8":
        model = ResNet(8, n_classes)
    elif net == "ResNet20":
        model = ResNet(20, n_classes)
    elif net == "ResNet32":
        model = ResNet(32, n_classes)
    elif net == "ResNet44":
        model = ResNet(44, n_classes)
    elif net == "ResNet56":
        model = ResNet(56, n_classes)
    else:
        raise ValueError("No such net: {}".format(net))

    model.apply(weights_init)

    return model


# for FedAws
def get_orth_weights(d, c):
    assert d > c, "d: {} must be larger than c: {}".format(d, c)
    xs = np.random.randn(d, d)
    pca = PCA(n_components=c)
    pca.fit(xs)

    # c \times d
    ws = pca.components_

    ws = torch.FloatTensor(ws)
    return ws


class ClassifyNet(nn.Module):
    def __init__(self, net, init_way, n_classes):
        super().__init__()
        self.net = net
        self.init_way = init_way
        self.n_classes = n_classes
        # self.ingraph = False

        model = get_basic_net(net, n_classes)

        self.h_size = model.h_size

        self.encoder = model.encoder

        self.classifier = nn.Linear(
            self.h_size, self.n_classes, bias=False
        )

        if self.init_way == "orth":
            ws = get_orth_weights(self.h_size, self.n_classes)
            self.classifier.load_state_dict({"weight": ws})

    # def __add__(self, other):
    #     if isinstance(other, int) and other == 0 : return self
    #     if not isinstance(other, ClassifyNet): raise TypeError
    #     return _model_add(self, other)
    #
    # def __radd__(self, other):
    #     return _model_add(self, other)
    #
    # def __sub__(self, other):
    #     if isinstance(other, int) and other == 0: return self
    #     if not isinstance(other, ClassifyNet): raise TypeError
    #     return _model_sub(self, other)
    #
    # def __mul__(self, other):
    #     return _model_scale(self, other)
    #
    # def __rmul__(self, other):
    #     return self*other
    #
    # def __truediv__(self, other):
    #     return self*(1.0/other)
    #
    # def __pow__(self, power, modulo=None):
    #     return _model_norm(self, power)
    #
    # def __neg__(self):
    #     return _model_scale(self, -1.0)
    #
    # def __sizeof__(self):
    #     if not hasattr(self, '__size'):
    #         param_size = 0
    #         param_sum = 0
    #         for param in self.parameters():
    #             param_size += param.nelement() * param.element_size()
    #             param_sum += param.nelement()
    #         buffer_size = 0
    #         buffer_sum = 0
    #         for buffer in self.buffers():
    #             buffer_size += buffer.nelement() * buffer.element_size()
    #             buffer_sum += buffer.nelement()
    #         self.__size = param_size + buffer_size
    #     return self.__size

    def forward(self, xs):
        hs = self.encoder(xs)
        logits = self.classifier(hs)
        return hs, logits

    # def norm(self, p=2):
    #     r"""
    #     Args:
    #         p (float): p-norm
    #
    #     Returns:
    #         the scale value of the p-norm of vectorized model parameters
    #     """
    #     return self**p
    #
    # def zeros_like(self):
    #     r"""
    #     Returns:
    #          a new model with the same architecture and all the parameters being set zero
    #     """
    #     return self*0
    #
    # def dot(self, other):
    #     r"""
    #     Args:
    #         other (Fmodule): the model with the same architecture of self
    #
    #     Returns:
    #         the dot value of the two vectorized models
    #     """
    #     return _model_dot(self, other)
    #
    # def cos_sim(self, other):
    #     r"""
    #     Args:
    #         other (Fmodule): the model with the same architecture of self
    #
    #     Returns:
    #         the cosine similarity value of the two vectorized models
    #     """
    #     return _model_cossim(self, other)
    #
    # def op_with_graph(self):
    #     self.ingraph = True
    #
    # def op_without_graph(self):
    #     self.ingraph = False
    #
    # def load(self, other):
    #     r"""
    #     Set the values of model parameters the same as the values of another model
    #     Args:
    #         other (Fmodule): the model with the same architecture of self
    #     """
    #     self.op_without_graph()
    #     self.load_state_dict(other.state_dict())
    #     return
    #
    # def freeze_grad(self):
    #     r"""
    #     All the gradients of the model parameters won't be computed after calling this method
    #     """
    #     for p in self.parameters():
    #         p.requires_grad = False
    #
    # def enable_grad(self):
    #     r"""
    #     All the gradients of the model parameters will be computed after calling this method
    #     """
    #     for p in self.parameters():
    #         p.requires_grad = True
    #
    # def zero_dict(self):
    #     r"""
    #     Set all the values of model parameters to be zero
    #     """
    #     self.op_without_graph()
    #     for p in self.parameters():
    #         p.data.zero_()
    #
    # def normalize(self):
    #     r"""
    #     Normalize the parameters of self to enable self.norm(2)=1
    #     """
    #     self.op_without_graph()
    #     self.load_state_dict((self/(self**2)).state_dict())
    #
    # def get_device(self):
    #     r"""
    #     Returns:
    #         the device of the tensors of this model
    #     """
    #     return next(self.parameters()).device
    #
    # def count_parameters(self, output=True):
    #     r"""
    #     Count the parameters for this model
    #
    #     Args:
    #         output (bool): whether to output the information to the stdin (i.e. console)
    #     Returns:
    #         the number of all the parameters in this model
    #     """
    #     # table = pt.PrettyTable(["Modules", "Parameters"])
    #     total_params = 0
    #     for name, parameter in self.named_parameters():
    #         if not parameter.requires_grad:
    #             # table.add_row([name, 0])
    #             continue
    #         params = parameter.numel()
    #         # table.add_row([name, params])
    #         total_params += params
    #     # if output:
    #     #     print(table)
    #     #     print(f"TotalTrainableParams: {total_params}")
    #     return total_params




class Reshape(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, xs):
        return xs.reshape((xs.shape[0], -1))


class MLPNet(nn.Module):
    def __init__(self, input_size, input_channel, n_classes=10):
        super().__init__()
        self.input_size = input_channel * input_size ** 2
        self.n_classes = n_classes

        self.encoder = nn.Sequential(
            Reshape(),
            nn.Linear(self.input_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.ReLU(True),
        )

        self.h_size = 128

        self.classifier = nn.Sequential(
            nn.Linear(128, n_classes)
        )

    def forward(self, xs):
        code = self.encoder(xs)
        logits = self.classifier(code)
        return code, logits


class LeNet(nn.Module):
    def __init__(self, input_size, input_channel, n_classes=10):
        super().__init__()
        self.input_size = input_size
        self.input_channel = input_channel
        self.n_classes = n_classes

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel, 16, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            Reshape(),
        )

        if self.input_size == 28:
            self.h_size = 16 * 4 * 4
        elif self.input_size == 32:
            self.h_size = 16 * 5 * 5
        else:
            raise ValueError("No such input_size.")

        self.classifier = nn.Sequential(
            nn.Linear(self.h_size, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, n_classes)
        )

    def forward(self, xs):
        code = self.encoder(xs)
        logits = self.classifier(code)
        return code, logits


class TFCNN(fmodule.FModule):
    def __init__(self, n_classes=10):
        super().__init__()
        self.n_classes = n_classes

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            Reshape(),
        )

        self.h_size = 64 * 4 * 4

        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(True),
            nn.Linear(128, n_classes)
        )

    def forward(self, xs):
        code = self.encoder(xs)
        logits = self.classifier(code)
        return code, logits


class VGG(nn.Module):
    def __init__(
        self,
        n_layer=11,
        n_classes=10,
        use_bn=False,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.n_classes = n_classes
        self.use_bn = use_bn

        self.cfg = self.get_vgg_cfg(n_layer)

        self.encoder = nn.Sequential(
            self.make_layers(self.cfg),
            Reshape(),
        )

        self.h_size = 512

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes),
        )

    def get_vgg_cfg(self, n_layer):
        if n_layer == 8:
            cfg = [
                64, 'M',
                128, 'M',
                256, 'M',
                512, 'M',
                512, 'M'
            ]
        elif n_layer == 11:
            cfg = [
                64, 'M',
                128, 'M',
                256, 256, 'M',
                512, 512, 'M',
                512, 512, 'M'
            ]
        elif n_layer == 13:
            cfg = [
                64, 64, 'M',
                128, 128, 'M',
                256, 256, 'M',
                512, 512, 'M',
                512, 512, 'M'
            ]
        elif n_layer == 16:
            cfg = [
                64, 64, 'M',
                128, 128, 'M',
                256, 256, 256, 'M',
                512, 512, 512, 'M',
                512, 512, 512, 'M'
            ]
        elif n_layer == 19:
            cfg = [
                64, 64, 'M',
                128, 128, 'M',
                256, 256, 256, 256, 'M',
                512, 512, 512, 512, 'M',
                512, 512, 512, 512, 'M'
            ]
        return cfg

    def conv3x3(self, in_channel, out_channel):
        layer = nn.Conv2d(
            in_channel, out_channel,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        return layer

    def make_layers(self, cfg, init_c=3):
        block = nn.ModuleList()

        in_c = init_c
        for e in cfg:
            if e == "M":
                block.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                block.append(self.conv3x3(in_c, e))
                if self.use_bn is True:
                    block.append(nn.BatchNorm2d(e))
                block.append(nn.ReLU(inplace=True))
                in_c = e
        block = nn.Sequential(*block)
        return block

    def forward(self, xs):
        hs = self.encoder(xs)
        logits = self.classifier(hs)
        return hs, logits


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes,
            kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(fmodule.FModule):
    """ 6n + 2: 8, 14, 20, 26, 32, 38, 44, 50, 56
    """

    def __init__(self, n_layer=20, n_classes=100):
        super().__init__()
        self.n_layer = n_layer
        self.n_classes = n_classes

        conv1 = nn.Conv2d(
            3, 16, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        bn1 = nn.BatchNorm2d(16)

        assert ((n_layer - 2) % 6 == 0), "SmallResNet depth is 6n+2"
        n = int((n_layer - 2) / 6)

        self.cfg = (BasicBlock, (n, n, n))
        self.in_planes = 16

        layer1 = self._make_layer(
            block=self.cfg[0], planes=16, stride=1, num_blocks=self.cfg[1][0],
        )
        layer2 = self._make_layer(
            block=self.cfg[0], planes=32, stride=2, num_blocks=self.cfg[1][1],
        )
        layer3 = self._make_layer(
            block=self.cfg[0], planes=64, stride=2, num_blocks=self.cfg[1][2],
        )

        avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.encoder = nn.Sequential(
            conv1,
            bn1,
            nn.ReLU(True),
            layer1,
            layer2,
            layer3,
            avgpool,
            Reshape(),
        )

        self.h_size = 64 * self.cfg[0].expansion

        self.classifier = nn.Linear(
            64 * self.cfg[0].expansion, n_classes
        )

    def _make_layer(self, block, planes, stride, num_blocks):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = block.expansion * planes
        return nn.Sequential(*layers)

    def forward(self, xs):
        hs = self.encoder(xs)
        logits = self.classifier(hs)
        return hs, logits

class ResNet18(fmodule.FModule):
    def __init__(self):
        super().__init__()
        resnet18 = torchvision.models.resnet18()
        resnet18.fc = nn.Linear(512, 10)
        resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

        resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

        resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
        resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

        resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
        resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

        resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
        resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
        self.model = resnet18

    def forward(self, x):
        return self.model(x)

class ModifiedResNet18(fmodule.FModule):
    def __init__(self, num_classes=10):
        super(ModifiedResNet18, self).__init__()
        # 初始化ResNet-18模型
        # self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 替换最后的全连接层，以适应新的类别数量
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
        # 除了全连接层外，获取所有层的输出
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)

        # 执行平均池化和扁平化操作
        x = self.resnet18.avgpool(x)
        flattened_representation = torch.flatten(x, 1)

        # 通过全连接层得到最终的分类结果
        classification_output = self.resnet18.fc(flattened_representation)

        # 返回一维表征和分类结果
        return flattened_representation, classification_output

class CNN_CIFAR10_FedAvg(fmodule.FModule):
    def __init__(self, n_classes=10):
        super(CNN_CIFAR10_FedAvg, self).__init__()
        self.input_require_shape = [3, -1, -1]
        self.target_require_shape = []

        self.name = 'CNN_CIFAR10_FedAvg'
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1600, 384),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(192, n_classes),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        return x, self.decoder(x)

# def normalize(m):
#     r"""
#     The new model that is the normalized version of the input model m=m/||m||_2
#
#     Args:
#         m (FModule): the model
#
#     Returns:
#         The new model that is the normalized version of the input model
#     """
#     return m/(m**2)
#
# def dot(m1, m2):
#     r"""
#     The dot value of the two models res = m1·m2
#
#     Args:
#         m1 (FModule): model 1
#         m2 (FModule): model 2
#
#     Returns:
#         The dot value of the two models
#     """
#     return m1.dot(m2)
#
# def cos_sim(m1, m2):
#     r"""
#     The cosine similarity value of the two models res=m1·m2/(||m1||*||m2||)
#
#     Args:
#         m1 (FModule): model 1
#         m2 (FModule): model 2
#
#     Returns:
#         The cosine similarity value of the two models
#     """
#     return m1.cos_sim(m2)
#
# def exp(m):
#     r"""
#     The element-wise res=exp(m) where all the model parameters satisfy mi=exp(mi)
#
#     Args:
#         m (FModule): the model
#
#     Returns:
#         The new model whose parameters satisfy mi=exp(mi)
#     """
#     return element_wise_func(m, torch.exp)
#
# def log(m):
#     r"""
#     The element-wise res=log(m) where all the model parameters satisfy mi=log(mi)
#
#     Args:
#         m (FModule): the model
#
#     Returns:
#         The new model whose parameters satisfy mi=log(mi)
#     """
#     return element_wise_func(m, torch.log)
#
# def element_wise_func(m, func):
#     r"""
#     The element-wise function on this model
#
#     Args:
#         m (FModule): the model
#         func: element-wise function
#
#     Returns:
#         The new model whose parameters satisfy mi=func(mi)
#     """
#     if m is None: return None
#     res = m.__class__().to(m.get_device())
#     if m.ingraph:
#         res.op_with_graph()
#         ml = get_module_from_model(m)
#         for md in ml:
#             rd = _modeldict_element_wise(md._parameters, func)
#             for l in md._parameters.keys():
#                 md._parameters[l] = rd[l]
#     else:
#         _modeldict_cp(res.state_dict(), _modeldict_element_wise(m.state_dict(), func))
#     return res
#
# def _model_to_tensor(m):
#     r"""
#     Convert the model parameters to torch.Tensor
#
#     Args:
#         m (FModule): the model
#
#     Returns:
#         The torch.Tensor of model parameters
#     """
#     return torch.cat([mi.data.view(-1) for mi in m.parameters()])
#
# def _model_from_tensor(mt, model_class):
#     r"""
#     Create model from torch.Tensor
#
#     Args:
#         mt (torch.Tensor): the tensor
#         model_class (FModule): the class defines the model architecture
#
#     Returns:
#         The new model created from tensors
#     """
#     res = model_class().to(mt.device)
#     cnt = 0
#     end = 0
#     with torch.no_grad():
#         for i, p in enumerate(res.parameters()):
#             beg = 0 if cnt == 0 else end
#             end = end + p.view(-1).size()[0]
#             p.data = mt[beg:end].contiguous().view(p.data.size())
#             cnt += 1
#     return res
#
# def _model_sum(ms):
#     r"""
#     Sum a list of models to a new one
#
#     Args:
#         ms (list): a list of models (i.e. each model's class is FModule(...))
#
#     Returns:
#         The new model that is the sum of models in ms
#     """
#     if len(ms)==0: return None
#     op_with_graph = sum([mi.ingraph for mi in ms]) > 0
#     res = ms[0].__class__().to(ms[0].get_device())
#     if op_with_graph:
#         mlks = [get_module_from_model(mi) for mi in ms]
#         mlr = get_module_from_model(res)
#         for n in range(len(mlr)):
#             mpks = [mlk[n]._parameters for mlk in mlks]
#             rd = _modeldict_sum(mpks)
#             for l in mlr[n]._parameters.keys():
#                 if mlr[n]._parameters[l] is None: continue
#                 mlr[n]._parameters[l] = rd[l]
#         res.op_with_graph()
#     else:
#         _modeldict_cp(res.state_dict(), _modeldict_sum([mi.state_dict() for mi in ms]))
#     return res
#
# def _model_average(ms = [], p = []):
#     r"""
#     Averaging a list of models to a new one
#
#     Args:
#         ms (list): a list of models (i.e. each model's class is FModule(...))
#         p (list): a list of real numbers that are the averaging weights
#
#     Returns:
#         The new model that is the weighted averaging of models in ms
#     """
#     if len(ms)==0: return None
#     if len(p)==0: p = [1.0 / len(ms) for _ in range(len(ms))]
#     op_with_graph = sum([w.ingraph for w in ms]) > 0
#     res = ms[0].__class__().to(ms[0].get_device())
#     if op_with_graph:
#         mlks = [get_module_from_model(mi) for mi in ms]
#         mlr = get_module_from_model(res)
#         for n in range(len(mlr)):
#             mpks = [mlk[n]._parameters for mlk in mlks]
#             rd = _modeldict_weighted_average(mpks, p)
#             for l in mlr[n]._parameters.keys():
#                 if mlr[n]._parameters[l] is None: continue
#                 mlr[n]._parameters[l] = rd[l]
#         res.op_with_graph()
#     else:
#         _modeldict_cp(res.state_dict(), _modeldict_weighted_average([mi.state_dict() for mi in ms], p))
#     return res
#
# def _model_add(m1, m2):
#     r"""
#     The sum of the two models m_new = m1+m2
#
#     Args:
#         m1 (FModule): model 1
#         m2 (FModule): model 2
#
#     Returns:
#         m_new = m1+m2
#     """
#     op_with_graph = m1.ingraph or m2.ingraph
#     res = m1.__class__().to(m1.get_device())
#     if op_with_graph:
#         res.op_with_graph()
#         ml1 = get_module_from_model(m1)
#         ml2 = get_module_from_model(m2)
#         mlr = get_module_from_model(res)
#         for n1, n2, nr in zip(ml1, ml2, mlr):
#             rd = _modeldict_add(n1._parameters, n2._parameters)
#             for l in nr._parameters.keys():
#                 if nr._parameters[l] is None: continue
#                 nr._parameters[l] = rd[l]
#     else:
#         _modeldict_cp(res.state_dict(), _modeldict_add(m1.state_dict(), m2.state_dict()))
#     return res
#
# def _model_sub(m1, m2):
#     r"""
#     The difference between the two models m_new = m1-m2
#
#     Args:
#         m1 (FModule): model 1
#         m2 (FModule): model 2
#
#     Returns:
#         m_new = m1-m2
#     """
#     op_with_graph = m1.ingraph or m2.ingraph
#     res = m1.__class__().to(m1.get_device())
#     if op_with_graph:
#         res.op_with_graph()
#         ml1 = get_module_from_model(m1)
#         ml2 = get_module_from_model(m2)
#         mlr = get_module_from_model(res)
#         for n1, n2, nr in zip(ml1, ml2, mlr):
#             rd = _modeldict_sub(n1._parameters, n2._parameters)
#             for l in nr._parameters.keys():
#                 if nr._parameters[l] is None: continue
#                 nr._parameters[l] = rd[l]
#     else:
#         _modeldict_cp(res.state_dict(), _modeldict_sub(m1.state_dict(), m2.state_dict()))
#     return res
#
# def _model_multiply(m1, m2):
#     r"""
#     Multiplying two models to obtain model m3 where m3[i] = m1[i] * m2[i] for each parameter i
#
#     Args:
#         m1 (FModule): model 1
#         m2 (FModule): model 2
#
#     Returns:
#         m3 = m1*m2
#     """
#     op_with_graph = m1.ingraph or m2.ingraph
#     res = m1.__class__().to(m1.get_device())
#     if op_with_graph:
#         res.op_with_graph()
#         ml1 = get_module_from_model(m1)
#         ml2 = get_module_from_model(m2)
#         mlr = get_module_from_model(res)
#         for n1, n2, nr in zip(ml1, ml2, mlr):
#             rd = _modeldict_multiply(n1._parameters, n2._parameters)
#             for l in nr._parameters.keys():
#                 if nr._parameters[l] is None: continue
#                 nr._parameters[l] = rd[l]
#     else:
#         _modeldict_cp(res.state_dict(), _modeldict_multiply(m1.state_dict(), m2.state_dict()))
#     return res
#
# def _model_divide(m1, m2):
#     r"""
#     Divide model1 by model2 to obtain model m3 where m3[i] = m1[i] / m2[i] for each parameter i
#
#     Args:
#         m1 (FModule): model 1
#         m2 (FModule): model 2
#
#     Returns:
#         m3 = m1/m2
#     """
#     op_with_graph = m1.ingraph or m2.ingraph
#     res = m1.__class__().to(m1.get_device())
#     if op_with_graph:
#         res.op_with_graph()
#         ml1 = get_module_from_model(m1)
#         ml2 = get_module_from_model(m2)
#         mlr = get_module_from_model(res)
#         for n1, n2, nr in zip(ml1, ml2, mlr):
#             rd = _modeldict_divide(n1._parameters, n2._parameters)
#             for l in nr._parameters.keys():
#                 if nr._parameters[l] is None: continue
#                 nr._parameters[l] = rd[l]
#     else:
#         _modeldict_cp(res.state_dict(), _modeldict_divide(m1.state_dict(), m2.state_dict()))
#     return res
#
# def _model_scale(m, s):
#     r"""
#     Scale a model's parameters by a real number
#
#     Args:
#         m (FModule): model
#         s (float|int): float number
#
#     Returns:
#         m_new = s*m
#     """
#     op_with_graph = m.ingraph
#     res = m.__class__().to(m.get_device())
#     if op_with_graph:
#         ml = get_module_from_model(m)
#         mlr = get_module_from_model(res)
#         res.op_with_graph()
#         for n, nr in zip(ml, mlr):
#             rd = _modeldict_scale(n._parameters, s)
#             for l in nr._parameters.keys():
#                 if nr._parameters[l] is None: continue
#                 nr._parameters[l] = rd[l]
#     else:
#         _modeldict_cp(res.state_dict(), _modeldict_scale(m.state_dict(), s))
#     return res
#
# def _model_norm(m, power=2):
#     r"""
#     Compute the norm of a model's parameters
#
#     Args:
#         m (FModule): model
#         power (float|int): power means the p in p-norm
#
#     Returns:
#         norm_p(model parameters)
#     """
#     op_with_graph = m.ingraph
#     res = torch.tensor(0.).to(m.get_device())
#     if op_with_graph:
#         ml = get_module_from_model(m)
#         for n in ml:
#             for l in n._parameters.keys():
#                 if n._parameters[l] is None: continue
#                 if n._parameters[l].dtype not in [torch.float, torch.float32, torch.float64]: continue
#                 res += torch.sum(torch.pow(n._parameters[l], power))
#         return torch.pow(res, 1.0 / power)
#     else:
#         return _modeldict_norm(m.state_dict(), power)
#
# def _model_dot(m1, m2):
#     r"""
#     The dot value of the two models res = m1·m2
#
#     Args:
#         m1 (FModule): model 1
#         m2 (FModule): model 2
#
#     Returns:
#         The dot value of the two models
#     """
#     op_with_graph = m1.ingraph or m2.ingraph
#     if op_with_graph:
#         res = torch.tensor(0.).to(m1.get_device())
#         ml1 = get_module_from_model(m1)
#         ml2 = get_module_from_model(m2)
#         for n1, n2 in zip(ml1, ml2):
#             res += _modeldict_dot(n1._parameters, n2._parameters)
#         return res
#     else:
#         return _modeldict_dot(m1.state_dict(), m2.state_dict())
#
# def _model_cossim(m1, m2):
#     r"""
#     The cosine similarity value of the two models res=m1·m2/(||m1||*||m2||)
#
#     Args:
#         m1 (FModule): model 1
#         m2 (FModule): model 2
#
#     Returns:
#         The cosine similarity value of the two models
#     """
#     op_with_graph = m1.ingraph or m2.ingraph
#     if op_with_graph:
#         res = torch.tensor(0.).to(m1.get_device())
#         ml1 = get_module_from_model(m1)
#         ml2 = get_module_from_model(m2)
#         l1 = torch.tensor(0.).to(m1.device)
#         l2 = torch.tensor(0.).to(m1.device)
#         for n1, n2 in zip(ml1, ml2):
#             res += _modeldict_dot(n1._parameters, n2._parameters)
#             for l in n1._parameters.keys():
#                 l1 += torch.sum(torch.pow(n1._parameters[l], 2))
#                 l2 += torch.sum(torch.pow(n2._parameters[l], 2))
#         return (res / torch.pow(l1, 0.5) * torch(l2, 0.5))
#     else:
#         return _modeldict_cossim(m1.state_dict(), m2.state_dict())
#
# def get_module_from_model(model, res = None):
#     r"""
#     Walk through all the sub modules of a model and return them as a list
#
#     Args:
#         model (FModule): model
#         res (None): should be remained None
#
#     Returns:
#         The list of all the sub-modules of a model
#     """
#     if res==None: res = []
#     ch_names = [item[0] for item in model.named_children()]
#     if ch_names==[]:
#         if model._parameters:
#             res.append(model)
#     else:
#         for name in ch_names:
#             get_module_from_model(model.__getattr__(name), res)
#     return res
#
# def _modeldict_cp(md1: dict, md2: dict):
#     r"""
#     Copy the values from the state_dict md2 to the state_dict md1
#
#     Args:
#         md1 (dict): the state_dict of a model
#         md2 (dict): the state_dict of a model
#     """
#     for layer in md1.keys():
#         md1[layer].data.copy_(md2[layer])
#     return
#
# def _modeldict_sum(mds):
#     r"""
#     Sum a list of modeldicts to a new one
#
#     Args:
#         mds (list): a list of modeldicts (i.e. each modeldict is the state_dict of a FModule(...))
#
#     Returns:
#         The new state_dict that is the sum of modeldicts in mds
#     """
#     if len(mds)==0: return None
#     md_sum = {}
#     for layer in mds[0].keys():
#         md_sum[layer] = torch.zeros_like(mds[0][layer])
#     for wid in range(len(mds)):
#         for layer in md_sum.keys():
#             if mds[0][layer] is None:
#                 md_sum[layer] = None
#                 continue
#             md_sum[layer] = md_sum[layer] + mds[wid][layer]
#     return md_sum
#
# def _modeldict_weighted_average(mds, weights=[]):
#     r"""
#     Averaging a list of modeldicts to a new one
#
#     Args:
#         mds (list): a list of modeldicts (i.e. the state_dict of models)
#         weights (list): a list of real numbers that are the averaging weights
#
#     Returns:
#         The new modeldict that is the weighted averaging of modeldicts in mds
#     """
#     if len(mds)==0:
#         return None
#     md_avg = {}
#     for layer in mds[0].keys(): md_avg[layer] = torch.zeros_like(mds[0][layer])
#     if len(weights) == 0: weights = [1.0 / len(mds) for _ in range(len(mds))]
#     for wid in range(len(mds)):
#         for layer in md_avg.keys():
#             if mds[0][layer] is None:
#                 md_avg[layer] = None
#                 continue
#             weight = weights[wid] if "num_batches_tracked" not in layer else 1
#             md_avg[layer] = md_avg[layer] + mds[wid][layer] * weight
#     return md_avg
#
# def _modeldict_to_device(md, device=None):
#     r"""
#     Transfer the tensors in a modeldict to the gpu device
#
#     Args:
#         md (dict): modeldict
#         device (torch.device): device
#     """
#     device = md[list(md)[0]].device if device is None else device
#     for layer in md.keys():
#         if md[layer] is None:
#             continue
#         md[layer] = md[layer].to(device)
#     return
#
# def _modeldict_to_cpu(md):
#     r"""
#     Transfer the tensors in a modeldict to the cpu memory
#
#     Args:
#         md (dict): modeldict
#     """
#     for layer in md.keys():
#         if md[layer] is None:
#             continue
#         md[layer] = md[layer].cpu()
#     return
#
# def _modeldict_zeroslike(md):
#     r"""
#     Create a modeldict that has the same shape with the input and all the values of it are zero
#
#     Args:
#         md (dict): modeldict
#
#     Returns:
#         a dict with the same shape and all the values are zero
#     """
#     res = {}
#     for layer in md.keys():
#         if md[layer] is None:
#             res[layer] = None
#             continue
#         res[layer] = md[layer] - md[layer]
#     return res
#
# def _modeldict_add(md1, md2):
#     r"""
#     The sum of the two modeldicts md3 = md1+md2
#
#     Args:
#         md1 (dict): modeldict 1
#         md2 (dict): modeldict 2
#
#     Returns:
#         a new model dict md3 = md1+md2
#     """
#     res = {}
#     for layer in md1.keys():
#         if md1[layer] is None:
#             res[layer] = None
#             continue
#         res[layer] = md1[layer] + md2[layer]
#     return res
#
# def _modeldict_scale(md, c):
#     r"""
#     Scale the tensors in a modeldict by a real number
#
#     Args:
#         md (dict): modeldict
#         c (float|int): a real number
#
#     Returns:
#         a new model dict md3 = c*md
#     """
#     res = {}
#     for layer in md.keys():
#         if md[layer] is None:
#             res[layer] = None
#             continue
#         res[layer] = md[layer] * c
#     return res
#
# def _modeldict_sub(md1, md2):
#     r"""
#     The difference of the two modeldicts md3 = md1-md2
#
#     Args:
#         md1 (dict): modeldict 1
#         md2 (dict): modeldict 2
#
#     Returns:
#         a new model dict md3 = md1-md2
#     """
#     res = {}
#     for layer in md1.keys():
#         if md1[layer] is None:
#             res[layer] = None
#             continue
#         res[layer] = md1[layer] - md2[layer]
#     return res
#
# def _modeldict_multiply(md1, md2):
#     r"""
#     Create a new modeldict md3 where md3[i]=md1[i]*md2[i] for each parameter i
#
#     Args:
#         md1 (dict): modeldict 1
#         md2 (dict): modeldict 2
#
#     Returns:
#         a new modeldict md3 = md1*md2
#     """
#     res = {}
#     for layer in md1.keys():
#         if md1[layer] is None:
#             res[layer] = None
#             continue
#         res[layer] = md1[layer] * md2[layer]
#     return res
#
# def _modeldict_divide(md1, md2):
#     r"""
#     Create a new modeldict md3 where md3[i]=md1[i]/md2[i] for each parameter i
#
#     Args:
#         md1 (dict): modeldict 1
#         md2 (dict): modeldict 2
#
#     Returns:
#         a new modeldict md3 = md1/md2
#     """
#     res = {}
#     for layer in md1.keys():
#         if md1[layer] is None:
#             res[layer] = None
#             continue
#         res[layer] = md1[layer]/md2[layer]
#     return res
#
# def _modeldict_norm(md, p=2):
#     r"""
#     The p-norm of the modeldict
#
#     Args:
#         md (dict): modeldict
#         p (float|int): a real number
#
#     Returns:
#         the norm of tensors in modeldict md
#     """
#     res = torch.tensor(0.).to(md[list(md)[0]].device)
#     for layer in md.keys():
#         if md[layer] is None: continue
#         if md[layer].dtype not in [torch.float, torch.float32, torch.float64]: continue
#         res += torch.sum(torch.pow(md[layer], p))
#     return torch.pow(res, 1.0/p)
#
# def _modeldict_to_tensor1D(md):
#     r"""
#     Cat all the tensors in the modeldict into a 1-D tensor
#
#     Args:
#         md (dict): modeldict
#
#     Returns:
#         a 1-D tensor that contains all the tensors in the modeldict
#     """
#     res = torch.Tensor().type_as(md[list(md)[0]]).to(md[list(md)[0]].device)
#     for layer in md.keys():
#         if md[layer] is None:
#             continue
#         res = torch.cat((res, md[layer].view(-1)))
#     return res
#
# def _modeldict_dot(md1, md2):
#     r"""
#     The dot value of the tensors in two modeldicts res = md1·md2
#
#     Args:
#         md1 (dict): modeldict 1
#         md2 (dict): modeldict 2
#
#     Returns:
#         The dot value of the two modeldicts
#     """
#     res = torch.tensor(0.).to(md1[list(md1)[0]].device)
#     for layer in md1.keys():
#         if md1[layer] is None:
#             continue
#         res += (md1[layer].view(-1).dot(md2[layer].view(-1)))
#     return res
#
# def _modeldict_cossim(md1, md2):
#     r"""
#     The cosine similarity value of the two models res=md1·md2/(||md1||*||md2||)
#
#     Args:
#         md1 (dict): modeldict 1
#         md2 (dict): modeldict 2
#
#     Returns:
#         The cosine similarity value of the two modeldicts
#     """
#     res = torch.tensor(0.).to(md1[list(md1)[0]].device)
#     l1 = torch.tensor(0.).to(md1[list(md1)[0]].device)
#     l2 = torch.tensor(0.).to(md1[list(md1)[0]].device)
#     for layer in md1.keys():
#         if md1[layer] is None:
#             continue
#         res += (md1[layer].view(-1).dot(md2[layer].view(-1)))
#         l1 += torch.sum(torch.pow(md1[layer], 2))
#         l2 += torch.sum(torch.pow(md2[layer], 2))
#     return res/(torch.pow(l1, 0.5)*torch.pow(l2, 0.5))
#
# def _modeldict_element_wise(md, func):
#     r"""
#     The element-wise function on the tensors of the modeldict
#
#     Args:
#         md (dict): modeldict
#         func: the element-wise function
#
#     Returns:
#         The new modeldict where the tensors in this dict satisfies mnew[i]=func(md[i])
#     """
#     res = {}
#     for layer in md.keys():
#         if md[layer] is None:
#             res[layer] = None
#             continue
#         res[layer] = func(md[layer])
#     return res
#
# def _modeldict_num_parameters(md):
#     r"""
#     The number of all the parameters in the modeldict
#
#     Args:
#         md (dict): modeldict
#
#     Returns:
#         The number of all the values of tensors in md
#     """
#     res = 0
#     for layer in md.keys():
#         if md[layer] is None: continue
#         s = 1
#         for l in md[layer].shape:
#             s *= l
#         res += s
#     return res
#
# def _modeldict_print(md):
#     r"""
#     Print the architecture of modeldict
#
#     Args:
#         md (dict): modeldict
#     """
#     for layer in md.keys():
#         if md[layer] is None:
#             continue
#         print("{}:{}".format(layer, md[layer]))
#
# def with_multi_gpus(func):
#     r"""
#     Decorate functions whose first parameter is model to carry out all the operations on the same device
#     """
#     def cal_on_personal_gpu(self, model, *args, **kargs):
#         origin_device = model.get_device()
#         # transfer to new device
#         new_args = []
#         new_kargs = {}
#         for arg in args:
#             narg = arg.to(self.device) if hasattr(arg, 'get_device') or hasattr(arg, 'device') else arg
#             new_args.append(narg)
#         for k,v in kargs.items():
#             nv = v.to(self.device) if hasattr(v, 'get_device') or hasattr(v, 'device') else v
#             new_kargs[k] = nv
#         model.to(self.device)
#         # calculating
#         res = func(self, model, *tuple(new_args), **new_kargs)
#         # transter to original device
#         model.to(origin_device)
#         if res is not None:
#             if type(res)==dict:
#                 for k,v in res.items():
#                     nv = v.to(origin_device) if hasattr(v, 'get_device') or hasattr(v, 'device') else v
#                     res[k] = nv
#             elif type(res)==tuple or type(res)==list:
#                 new_res = []
#                 for v in res:
#                     nv = v.to(origin_device) if hasattr(v, 'get_device') or hasattr(v, 'device') else v
#                     new_res.append(nv)
#                 if type(res)==tuple:
#                     res = tuple(new_res)
#             else:
#                 res = res.to(origin_device) if hasattr(res, 'get_device') or hasattr(res, 'device') else res
#         return res
#     return cal_on_personal_gpu