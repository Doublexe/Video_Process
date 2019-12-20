from .ResNet_IFN_aae_s1 import resnet50_IFN as resnet50_IFN_aae
from torchvision import transforms
import torch
from PIL import Image

def load_state_dict(network, save_path):
    checkpoint = torch.load(save_path)
    network.load_state_dict(checkpoint)
    return network


class AAE(object):
    def __init__(self, model_pth, device):
        self.data_transforms = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        model = resnet50_IFN_aae(num_classes=18530)
        self.device = device
        self.model = load_state_dict(model, model_pth)
        self.model.to(device)
        self.model.eval()

    def __call__(self, x):
        xs = []
        for x_ in x:
            x_ = Image.fromarray(x_)
            x_ = self.data_transforms(x_)
            xs.append(x_)
        x = torch.stack(xs)
        if self.device:
            x = x.to(self.device)
        f = self.model(x)
        return f.detach().cpu().numpy()
