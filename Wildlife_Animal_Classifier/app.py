import torch
import gradio as gr
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
from torch.nn import init
import torchvision.transforms as transforms
from PIL import Image


# MobileNetV3 Model Definition (keep this exactly as in your original code)
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
        x = self.bneck(x)          # Execute all bottleneck blocks
        x = self.hs2(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.hs3(self.bn3(self.linear3(x)))
        return self.linear4(x)


model = MobileNetV3_Small().to('cuda')
device = torch.device("cuda")

# State dict is stored relative to the models folder
state_dict_path = os.path.abspath(os.path.join(os.getcwd(), "models", "MobileNet3_small_StateDictionary.pth"))
model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
model.eval()




# Class Labels
classes = [
    'antelope', 'buffalo', 'chimpanzee', 'cow', 'deer', 'dolphin',
    'elephant', 'fox', 'giant+panda', 'giraffe', 'gorilla', 'grizzlybear',
    'hamster', 'hippopotamus', 'horse', 'humpbackwhale', 'leopard', 'lion',
    'moose', 'otter', 'ox', 'pig', 'polarbear', 'rabbit', 'rhinoceros',
    'seal', 'sheep', 'squirrel', 'tiger', 'zebra'
]

# Precompute example image paths
example_dir = "test_images"
example_images = [os.path.join(example_dir, f) for f in os.listdir(example_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

# PREPROCESSING PIPELINE (ADD THIS BACK)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
example_images = [os.path.join(example_dir, f) for f in os.listdir(example_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

def predict(img_path):
    """Process single image and return prediction"""
    if not img_path:
        return "Please select or upload an image first"
    
    try:
        image = Image.open(img_path).convert('RGB')
        tensor = preprocess(image).unsqueeze(0).to(device)
        
        with torch.inference_mode():
            outputs = model(tensor)
            _, pred = torch.max(outputs, 1)
        
        return classes[pred.item()]
    
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks(title="Wildlife Animal Classifier", css="") as demo:
    gr.Markdown("## 🐾 Wildlife Animal Classifier")
    gr.Markdown("Select an image below or upload your own, then click Classify")
    gr.Markdown("Trained Classes: antelope, buffalo, chimpanzee, cow, deer, dolphin, elephant, fox, giantpanda, giraffe, gorilla, grizzlybear, hamster, hippopotamus, horse, humpbackwhale, leopard, lion, moose, otter, ox, pig, polarbear, rabbit, rhinoceros, seal, sheep, squirrel, tiger, zebra")
    # Store current image path
    current_image = gr.State()
    
    with gr.Row():
        with gr.Column():
            image_preview = gr.Image(label="Selected Image", type="filepath")
            upload_btn = gr.UploadButton("Upload Custom Image", file_types=["image"])
            classify_btn = gr.Button("Classify 🚀", variant="primary")
        result = gr.Textbox(label="Prediction", lines=3)
    
    # Example gallery at bottom
    with gr.Row(variant="panel"):
        examples_gallery = gr.Gallery(
            value=example_images,
            label="Example Images (Click to Select)",
            columns=7,        
            elem_id="my_media_gallery",
            allow_preview=False,
            elem_classes=["centered-examples"]
        )

    # Handle image selection from examples - FIXED OUTPUTS
    def select_example(evt: gr.SelectData):
        selected_path = example_images[evt.index]
        return selected_path, selected_path  # Return both image preview and state
    
    examples_gallery.select(
        fn=select_example,
        outputs=[image_preview, current_image],  # Match both components
        show_progress=False
    )
    
    # Fix upload handler too
    upload_btn.upload(
        fn=lambda file: (file.name, file.name),  # Return both preview and state
        inputs=upload_btn,
        outputs=[image_preview, current_image]
    )

    # Handle classification
    classify_btn.click(
        fn=predict,
        inputs=current_image,
        outputs=result
    )

if __name__ == "__main__":
    demo.launch(show_error=True)