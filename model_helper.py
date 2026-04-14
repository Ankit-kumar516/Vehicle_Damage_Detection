import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

trained_model = None
class_names = ['Front Breakage', 'Front Crushed', 'Front Normal', 'Rear Breakage', 'Rear Crushed', 'Rear Normal']
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "saved_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Load the pre-trained ResNet model
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        # Freeze all layers except the final fully connected layer
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer4 and fc layers
        for param in self.model.layer4.parameters():
            param.requires_grad = True

            # Replace the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x


def _load_model():
    global trained_model

    if trained_model is not None:
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    trained_model = CarClassifierResNet(num_classes=len(class_names))
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    trained_model.load_state_dict(state_dict)
    trained_model.to(DEVICE)
    trained_model.eval()


def predict(image):
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        raise TypeError("image must be a file path or PIL.Image.Image")

    image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    _load_model()

    with torch.no_grad():
        output = trained_model(image_tensor)
        probabilities = torch.softmax(output, dim=1).squeeze(0)
        confidence, predicted_class = torch.max(probabilities, 0)

        return {
            "label": class_names[predicted_class.item()],
            "confidence": float(confidence.item()),
            "probabilities": {
                class_names[idx]: float(prob.item())
                for idx, prob in enumerate(probabilities)
            }
        }
