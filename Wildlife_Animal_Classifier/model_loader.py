import torch
import gradio as gr
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision.transforms as transforms
from PIL import Image
import os

# MobileNetV3 Model Definition
class hswish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3) / 6

class hsigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3) / 6

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size//reduction, 1, bias=False),
            nn.BatchNorm2d(in_size//reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size//reduction, in_size, 1, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class Block(nn.Module):
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super().__init__()
        self.stride = stride
        self.se = semodule
        self.conv1 = nn.Conv2d(in_size, expand_size, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size, stride, kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se: out = self.se(out)
        return out + self.shortcut(x) if self.stride==1 else out

class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=30):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()
        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(), SeModule(16), 2),
            Block(3, 16, 72, 24, nn.ReLU(), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(), None, 1),
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        )
        self.conv2 = nn.Conv2d(96, 576, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(576, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None: init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.hs1(self.bn1(self.conv1(x)))
        x = self.bneck(x)          # Run all 11 bottleneck blocks
        x = self.hs2(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.hs3(self.bn3(self.linear3(x)))
        return self.linear4(x)

"""!!!!Change the paths to the image you want to predict and the the model save location!!!!"""

# Initialize Model using the provided Dict
user_device = "cuda" if torch.cuda.is_available() else "cpu"
model = MobileNetV3_Small()
# Load state dict with relative paths
state_dict = torch.load(os.path.join(os.getcwd(), "MobileNet3_small_StateDictionary.pth"), map_location=user_device)
model.load_state_dict(state_dict)
model.eval()

# Class Labels that can be detected
classes = [
    'antelope', 'buffalo', 'chimpanzee', 'cow', 'deer', 'dolphin',
    'elephant', 'fox', 'giant+panda', 'giraffe', 'gorilla', 'grizzlybear',
    'hamster', 'hippopotamus', 'horse', 'humpbackwhale', 'leopard', 'lion',
    'moose', 'otter', 'ox', 'pig', 'polarbear', 'rabbit', 'rhinoceros',
    'seal', 'sheep', 'squirrel', 'tiger', 'zebra'
]


# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Predict Image Classes
def predict(imagepath:str):
    """Process one image and returns prediction"""
    img = Image.open(imagepath)
    img = preprocess(img)
    with torch.inference_mode():
        outputs = model(img.unsqueeze(dim=0)) # because the model was trained on batches
        preds = outputs.argmax(dim=1)
    print(f"The image shows the class: {classes[preds]}")    