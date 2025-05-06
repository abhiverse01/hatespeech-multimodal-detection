import streamlit as st 
st.set_page_config(page_title="Multimodal Hate Speech Detection")

from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from transformers import BertTokenizer, BertModel
import numpy as np
import os

# ====== Configuration ======
MODEL_PATH = "best_model_weights.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["Non-Hate", "Religion", "Racist", "Homophobic", "Sexist", "Abusive"]

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
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.img_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(768 + 2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(LABELS))  # Output for 6 labels
        )

    def forward(self, input_ids, attention_mask, image):
        text_feat = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]  # CLS token

        img_feat = self.img_encoder(image).squeeze(-1).squeeze(-1)
        combined = torch.cat((text_feat, img_feat), dim=1)
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
st.write("Upload a **meme image** and provide its **caption** to detect potential hate speech.")

with st.expander("ℹ️ About the Labels"):
    st.markdown("""
    - **Non-Hate**: Neutral or non-offensive content  
    - **Religion**: Offensive content targeting religion  
    - **Racist**: Racial or ethnic hate speech  
    - **Homophobic**: Anti-LGBTQ+ content  
    - **Sexist**: Gender or sexist content  
    - **Abusive**: General profanity or abuse  
    """)

uploaded_image = st.file_uploader("📷 Upload Meme Image", type=["jpg", "jpeg", "png"])
input_text = st.text_area("📝 Enter Meme Caption")

threshold = st.slider("⚙️ Confidence Threshold", min_value=0.30, max_value=1.0, value=0.5, step=0.05)

if threshold < 0.4:
    st.warning("⚠️ Low thresholds may cause false positives. Consider using at least 0.4 for reliable detection.")

if uploaded_image and input_text:
    try:
        # ===== Preprocessing =====
        image = Image.open(uploaded_image).convert("RGB")
        image_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

        encoded = tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)

        # ===== Model Prediction =====
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, image_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]  # shape: [6]

        # ===== Display Image =====
        st.image(image, caption="🖼️ Uploaded Meme", use_container_width=True)
        st.markdown("---")

        # ===== Format and Show Predictions =====
        st.subheader("🔍 Prediction Scores")
        detected_labels = []

        for i, (label, prob) in enumerate(zip(LABELS, probs)):
            prob = float(prob)
            st.markdown(f"**{label}**: `{prob:.2f}`")
            st.progress(prob)

            if i != 0 and prob >= threshold:
                detected_labels.append((label, prob))

        st.markdown("---")

        if detected_labels:
            st.error("⚠️ **Hate speech detected in this meme.**")
            st.subheader("🚨 Detected Hate Categories:")
            for label, score in detected_labels:
                st.markdown(f"- **{label}** (confidence `{score:.2f}`)")
        else:
            st.success("✅ **This meme does not contain hate speech.**")
            st.caption("All hate-related scores are below the selected threshold.")

    except Exception as e:
        st.error(f"❌ An error occurred during processing: `{str(e)}`")
