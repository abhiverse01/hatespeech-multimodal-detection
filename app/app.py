import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from transformers import BertTokenizer, BertModel
import numpy as np
import os

# ====== Configuration ======
MODEL_PATH = "model_weights.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Load Tokenizer and Transforms ======
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ====== Define the Model Class ======
class HateSpeechClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        
        # ✅ Use torchvision instead of torch.hub (offline safe)
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.img_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        self.classifier = nn.Sequential(
            nn.Linear(768 + 2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 3)
        )

    def forward(self, input_ids, attention_mask, image):
        text_feat = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]  # [CLS] token output

        img_feat = self.img_encoder(image).squeeze(-1).squeeze(-1)  # ResNet output
        combined = torch.cat((text_feat, img_feat), dim=1)  # Concatenate text + image
        return self.classifier(combined)

# ====== Load Model ======
@st.cache_resource
def load_model():
    model = HateSpeechClassifier().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# ====== Streamlit UI ======
st.title("🧠 Multimodal Hate Speech Detection")
st.write("Upload a **meme image** and provide a **caption** to classify if it contains hate speech.")

uploaded_image = st.file_uploader("📷 Upload a meme image...", type=["jpg", "jpeg", "png"])
input_text = st.text_area("📝 Enter the meme caption text here:")

if uploaded_image and input_text:
    try:
        # Image preprocessing
        image = Image.open(uploaded_image).convert("RGB")
        image_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

        # Text preprocessing
        encoded = tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)

        # Model prediction
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, image_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]  # Multi-label probabilities

        # Display results
        st.image(image, caption="Uploaded Meme", use_column_width=True)
        st.write("🔍 **Prediction Scores (0-1):**", {f"Label {i}": float(f"{p:.2f}") for i, p in enumerate(probs)})

        if (probs > 0.5).any():
            st.error("⚠️ Hate speech detected in this meme.")
        else:
            st.success("✅ This meme does not contain hate speech.")

    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
