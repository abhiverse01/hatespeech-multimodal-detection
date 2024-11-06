# Multi-Modal Hate Speech Detection using Deep Learning.

# Dataset:

- Creating a hate speech detection system using deep learning requires datasets encompassing a wide range of text, audio, and images (if applicable) to understand and detect hate speech across multiple modalities. Here are some common datasets often used for multi-modal hate speech detection:

1. **HateXplain**:  
   - A multi-modal dataset for hate speech detection includes annotated data with rationales to justify why certain content is classified as hate speech.
   - It includes textual data and associated images, making it suitable for multi-modal models.
   - [Dataset link](https://github.com/hate-alert/HateXplain)

2. **MMHS150K (Multi-Modal Hate Speech 150K)**:  
   - Contains 150,000 tweets, each with an associated image and labels for offensive and hate speech.
   - Focused on Twitter data, this dataset captures real-world examples of hate speech in a multi-modal format.
   - [Dataset link](https://github.com/FirojIslam/Multimodal-Hate-Speech)

3. **CAiRE (HateMeme)**:  
   - A dataset specifically designed for detecting hate speech in memes.
   - It includes images and accompanying text, making it suitable for image-text multi-modal deep learning models.
   - [Dataset link](https://ai.facebook.com/datasets/hateful-memes/)

4. **CONAN**:  
   - A synthetic hate speech dataset covering various languages, which makes it beneficial for training models across different linguistic contexts.
   - The dataset includes text-based data but can be paired with images from other datasets to create a multi-modal input pipeline.
   - [Dataset link](https://github.com/uds-lsv/conan)

5. **HASOC (Hate Speech and Offensive Content Identification)**:  
   - Focused on identifying hate speech in social media posts across multiple languages, including English, Hindi, and German.
   - Although it is primarily a text-based dataset, it can be augmented with additional modalities for multi-modal tasks.
   - [Dataset link](https://hasocfire.github.io/hasoc/2020/dataset.html)

6. **Flickr and Twitter Datasets**:  
   - These datasets contain image-caption pairs, which can be useful for training models to detect hate speech that appears alongside imagery, such as memes or annotated images.
   - While not specifically focused on hate speech, you can filter and annotate the data for relevant hate-related content.
   - [Flickr Dataset link](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)

7. **AMIT (Aggressive and Offensive Multimodal Image and Text Dataset)**:  
   - Designed specifically for identifying aggressive and offensive content in a multi-modal context.
   - Contains images and text pairs commonly found on social media and annotated for hate speech detection tasks.

These datasets should be accompanied by pre-processing steps, especially when dealing with images and text separately, before feeding them into a deep learning model such as a BERT+CNN or a Vision Transformer (ViT) setup for multi-modal classification tasks.
