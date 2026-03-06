from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_EMBED_DIM = 512

# ----- Image encoder -----
class ImageEncoder(nn.Module):
    def __init__(self, out_dim=IMAGE_EMBED_DIM):
        super().__init__()
        # using ResNet50
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1] # remove classifier
        self.backbone = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(resnet.fc.in_features, out_dim)


    def forward(self, x):
        feats = self.backbone(x)
        feats = feats.view(feats.size(0), -1)
        out = self.fc(feats)
        return out

_image_transform = transforms.Compose([
transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])



def extract_image_feature(image_path: Path, model: ImageEncoder, device=device):
    img = Image.open(image_path).convert('RGB')
    x = _image_transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        emb = model(x)
        return emb.cpu().numpy().squeeze()

if __name__ == "__main__":
    # Example usage
    model = ImageEncoder().to(device)
    image_path = Path("dataset\\images\\373947\\1.jpg")  # Replace with your image path
    feature = extract_image_feature(image_path, model)
    print("Extracted image feature shape:", feature)