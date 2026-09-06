from pathlib import Path

import joblib
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT / "best_resnet18_finetuned_cardiomegaly.pt"
)

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

FUSION_ALPHA = 0.80


class CardiomegalyPredictor:
    def __init__(self):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.image_model = self._load_image_model()

        self.vectorizer = joblib.load(
            ARTIFACT_DIR / "tfidf_vectorizer.joblib"
        )

        self.text_model = joblib.load(
            ARTIFACT_DIR / "text_classifier.joblib"
        )

        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _load_image_model(self):
        model = resnet18(weights=None)

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1)

        model.load_state_dict(
            torch.load(
                CHECKPOINT_PATH,
                map_location=self.device,
            )
        )

        model = model.to(self.device)
        model.eval()

        return model

    def predict_image(self, image_path):
        image = Image.open(image_path).convert("RGB")

        image_tensor = self.image_transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.image_model(image_tensor)
            probability = torch.sigmoid(logit).item()

        return probability

    def predict_text(self, indication):
        X = self.vectorizer.transform([indication])

        probability = float(
            self.text_model.predict_proba(X)[0, 1]
        )

        return probability

    def predict(self, image_path, indication):
        image_probability = self.predict_image(image_path)
        text_probability = self.predict_text(indication)

        fusion_probability = (
            FUSION_ALPHA * image_probability
            + (1 - FUSION_ALPHA) * text_probability
        )

        return {
            "image_probability": image_probability,
            "text_probability": text_probability,
            "fusion_probability": fusion_probability,
            "fusion_alpha": FUSION_ALPHA,
        }