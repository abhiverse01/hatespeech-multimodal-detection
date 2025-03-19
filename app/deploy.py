from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torchvision import models, transforms
import torch
from PIL import Image
import os
from huggingface_hub import HfApi, HfFolder, Repository

# Load your text model and tokenizer
text_model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModelForSequenceClassification.from_pretrained(text_model_name)

# Load your image model
image_model = models.resnet50(pretrained=True)
image_model.fc = torch.nn.Linear(image_model.fc.in_features, 2)  # Binary classification

# Create a pipeline for text classification
text_classifier = pipeline("text-classification", model=text_model, tokenizer=tokenizer)

# Function to preprocess images
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    return preprocess(image).unsqueeze(0)

# Function to classify text and image
def classify_text_and_image(text, image_path):
    text_result = text_classifier(text)[0]
    image_tensor = preprocess_image(image_path)
    image_result = image_model(image_tensor)
    image_result = torch.nn.functional.softmax(image_result, dim=1)
    return text_result, image_result

# Function to deploy the model
def deploy_model(api_token):

    # Save the API token
    HfFolder.save_token(api_token)

    # Initialize the HfApi
    api = HfApi()

    # Create a new repository
    repo_name = ""
    repo_url = api.create_repo(repo_name, private=True)

    # Clone the repository
    repo = Repository(local_dir=repo_name, clone_from=repo_url)

    # Save the models and tokenizer to the repository
    text_model.save_pretrained(repo_name)
    tokenizer.save_pretrained(repo_name)
    torch.save(image_model.state_dict(), os.path.join(repo_name, "resnet50.pth"))

    # Push the repository to Hugging Face
    repo.push_to_hub()

# Replace 'your-hugging-face-api-token' with your actual Hugging Face API token
api_token = ""
deploy_model(api_token)