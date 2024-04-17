import torch
import torch.nn as nn
from torchvision import models

class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModifiedResNet18, self).__init__()
        # 初始化ResNet-18模型
        self.resnet18 = models.resnet18(pretrained=False)

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

if __name__ == "__main__":
    # 创建模型实例
    model = ModifiedResNet18(num_classes=1000)

    # 假设有一个输入张量
    input_tensor = torch.randn(1, 3, 224, 224)  # 示例输入，大小为[batch_size, channels, height, width]

    # 获取一维表征和分类结果
    flattened_representation, classification_output = model(input_tensor)

    # 打印输出的维度
    print(flattened_representation.shape, classification_output.shape)