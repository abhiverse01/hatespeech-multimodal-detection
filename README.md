# Multi-Modal Hate Speech Detection using Deep Learning.

---

#### **Introduction**

This phase of the project focuses on creating a **multi-modal dataset** for hate speech detection, combining **text** (tweets) and **images** (associated with the tweets). The goal is to prepare a clean, structured dataset that can be used to train a deep learning model capable of detecting hate speech in multi-modal content.  
The dataset preparation involves these key phases:

1. **Data Downloading and Image Loading**.
2. **Text Cleaning and Preprocessing**.
3. **Image Preprocessing**.
4. **Multi-Modal Dataset Creation**.
5. **DataLoader Preparation**.

---

### **Phase 1: Data Downloading and Image Loading**

#### **1.1 Data Downloading Function**

- **Objective**: Download the dataset and extract relevant information (text, image URLs, and labels).
- **Process**:
  - The dataset is loaded from a JSON file (`MMHS150K_GT.json`).
  - Each row contains:
    - `tweet_text`: The text content of the tweet.
    - `img_url`: The URL of the associated image.
    - `labels`: Multi-label classification for hate speech (e.g., `[4, 1, 3]` corresponds to `[Religion, Racist, Homophobe]`).

#### **1.2 Image Loading**

- **Objective**: Download images from URLs and save them locally.
- **Process**:
  - A function (`load_dataset`) is created to:
    - Iterate through the dataset rows.
    - Download images from the `img_url` column.
    - Save images locally in a folder (`dataset/images/`) with numeric filenames (e.g., `0.jpg`, `1.jpg`).
    - Skip rows where the image cannot be downloaded (e.g., broken links or network errors).
  - **Key Details**:
    - If an image fails to download, the entire row is skipped to ensure **text-image matching**.
    - Numeric filenames are used to avoid issues with special characters or spaces in filenames.

#### **1.3 Challenges Faced**

- **Broken Image URLs**: Some image URLs were inaccessible, leading to skipped rows.
- **Network Errors**: The function skips problematic rows and ensures only valid text-image pairs are included in the final dataset.

---

### **Phase 2: Data Loading and Initial Exploration**

#### **2.1 Loading the First 50 Samples**

- **Objective**: Test the data loading and preprocessing pipeline on a small subset of the dataset.
- **Process**:
  - The first 50 samples are loaded and processed.
  - Images are downloaded, and text and labels are extracted.
  - The dataset is saved as `dataset.csv`.

#### **2.2 Key Details**

- **Dataset Structure**:
  - `tweet_text`: The text content of the tweet.
  - `image_path`: Local path to the downloaded image.
  - `labels`: Multi-label classification for hate speech.

#### **2.3 Challenges Faced**

- **Inconsistent Data**: Some rows had missing or invalid data (e.g., empty text or labels).
- **Solution**: Rows with missing or invalid data were skipped during preprocessing.

---

### **Phase 3: Text Cleaning and Preprocessing**

#### **3.1 Text Cleaning**

- **Objective**: Clean the text data to remove noise and prepare it for tokenization.
- **Process**:
  - **Lowercasing**: Convert all text to lowercase.
  - **Remove URLs**: Remove any URLs from the text.
  - **Remove User Mentions**: Remove Twitter handles (e.g., `@username`).
  - **Remove Special Characters**: Remove punctuation, emojis, and other special characters.
  - **Remove Extra Whitespace**: Trim extra spaces and newlines.

#### **3.2 Tokenization**

- **Objective**: Convert cleaned text into tokenized input for BERT.
- **Process**:
  - Use the `BertTokenizer` from Hugging Face to tokenize the text.
  - Generate:
    - `input_ids`: Tokenized text converted to numerical IDs.
    - `attention_mask`: Mask to indicate which tokens are actual words and which are padding.

#### **3.3 Key Details**

- **Tokenization Parameters**:
  - `max_length=128`: Pad/truncate text to 128 tokens.
  - `padding="max_length"`: Pad shorter texts to the maximum length.
  - `truncation=True`: Truncate longer texts to the maximum length.

#### **3.4 Challenges Faced**

- **Text Length Variability**: Some tweets were longer than 128 tokens.
- **Solution**: Truncate longer texts to fit the model’s input size.

---

### **Phase 4: Image Preprocessing**

#### **4.1 Image Transformation**

- **Objective**: Preprocess images for input into a ResNet-50 model.
- **Process**:
  - **Resize**: Resize images to `224x224` pixels.
  - **Normalize**: Normalize pixel values using ImageNet statistics (`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`).
  - **Convert to Tensor**: Convert images to PyTorch tensors.

#### **4.2 Save Transformed Images**

- **Objective**: Save preprocessed images for later use.
- **Process**:
  - Save transformed images as `.pt` files in a folder (`dataset/transformed_images/`).
  - Update the dataset to include paths to the transformed images (`transformed_image_path`).

#### **4.3 Key Details**

- **Image Shape**: Transformed images have the shape `[3, 224, 224]` (3 color channels, 224x224 resolution).
- **Normalization**: Pixel values are normalized to the range `[-3, 3]` after transformation.

#### **4.4 Challenges Faced**

- **Image Loading Errors**: Some images failed to load or transform.
- **Solution**: Use a black placeholder image for failed transformations.

---

### **Phase 5: DataLoader Preparation**

The **DataLoader** is a PyTorch utility that handles **batching**, **shuffling**, and **loading** of the multi-modal dataset. It takes the `MultiModal Dataset` and prepares it for training by organizing the data into batches and providing an iterator over the dataset.

#### **5.1 Multi-Modal Dataset Creation**

- **Objective**: Combine text, images, and labels into a PyTorch `Dataset`.
- **Process**:
  - Create a custom `MultiModalDataset` class that:
    - Loads tokenized text (`input_ids`, `attention_mask`).
    - Loads transformed images from `.pt` files.
    - Loads labels as tensors.

#### **5.2 DataLoader Creation**

- **Objective**: Create a `DataLoader` to handle batching and shuffling.
- **Process**:
  - Use the `DataLoader` class to:
    - Batch the dataset into groups of 32 samples.
    - Shuffle the data to improve training.
    - Provide an iterator for looping through the dataset.

#### **5.3 Key Details**

- **Batch Size**: 32 samples per batch.
- **Shuffling**: Enabled to randomize the order of samples.
- **Output Shapes**:
  - `input_ids`: `[32, 128]` (32 samples, 128 tokens each).
  - `attention_mask`: `[32, 128]` (32 samples, 128 tokens each).
  - `images`: `[32, 3, 224, 224]` (32 images, 3 channels, 224x224 resolution).
  - `labels`: `[32, 3]` (32 samples, 3 labels each).

**5.6 Final Multi-Modal Dataset Prepared**  
The DataLoader used is a runtime object that loads data from the dataset for model training.

- The **dataset** (text, images, and labels) is saved in:
  - dataset_transformed.csv (text and labels).
  - dataset/transformed_images/ (transformed images as .pt files).

### **\# Columns in `dataset_transformed.csv`**

#### **1\. `tweet_text`**

- **Description**: The text content of the tweet.
- **Preprocessing**:
  - Cleaned to remove noise (e.g., URLs, user mentions, special characters).
  - Converted to lowercase.
- **Purpose**: Used as an input for the text-based model (e.g., BERT).

#### **2\. `image_path`**

- **Description**: The local path to the original downloaded image (e.g., `dataset/images/0.jpg`).
- **Preprocessing**:
  - Images are downloaded from URLs and saved locally.
  - Numeric filenames are used to avoid issues with special characters or spaces.
- **Purpose**: Provides a reference to the original image file.

#### **3\. `labels`**

- **Description**: Multi-label classification for hate speech (e.g., `[4, 1, 3]` corresponds to `[Religion, Racist, Homophobe]`).
- **Preprocessing**:
  - Converted from string representation (e.g., `"[4, 1, 3]"`) to a list of integers using `ast.literal_eval`.
- **Purpose**: Used as the target variable for training the model.

#### **4\. `input_ids`**

- **Description**: Tokenized text converted to numerical IDs for input into BERT.
- **Preprocessing**:
  - Generated using the `BertTokenizer`.
  - Padded/truncated to a fixed length of 128 tokens.
- **Purpose**: Represents the tokenized text for the text-based model.

#### **5\. `attention_mask`**

- **Description**: Mask to indicate which tokens are actual words and which are padding.
- **Preprocessing**:
  - Generated alongside `input_ids` using the `BertTokenizer`.
  - Contains `1` for actual tokens and `0` for padding tokens.
- **Purpose**: Helps the model ignore padding tokens during training.

#### **6\. `transformed_image_path`**

- **Description**: The local path to the preprocessed image (e.g., `dataset/transformed_images/0.pt`).
- **Preprocessing**:
  - Images are resized to `224x224` pixels.
  - Normalized using ImageNet statistics.
  - Converted to PyTorch tensors and saved as `.pt` files.
- **Purpose**: Provides a reference to the preprocessed image tensor for the image-based model (i.e., ResNet-50).


### **Phase 6: Embeddings Generation**  

#### **6.1 Overview**  
To improve multi-modal learning, we generate **text, image, and multimodal embeddings**. These embeddings represent the underlying semantic information in a numerical format, enabling deep learning models to detect patterns more effectively.  

#### **6.2 Text Embeddings**  
- **Model Used**: `BERT-based transformer`  
- **Process**:
  - Tokenized tweet texts are passed through a pre-trained BERT model.
  - The `[CLS]` token representation is extracted as the **text embedding**.
  - The resulting **vector size** is `768`.  
- **Usage**:
  - Captures contextual meaning of tweets.
  - Used as input to the multi-modal classifier.  

#### **6.3 Image Embeddings**  
- **Model Used**: `ResNet-50 (pre-trained on ImageNet)`  
- **Process**:
  - Preprocessed images are fed into a ResNet-50 model.
  - The **final layer before classification** is extracted as the **image embedding**.
  - The resulting **vector size** is `2048`.  
- **Usage**:
  - Captures high-level visual features from images.
  - Helps in detecting hate symbols or offensive visual elements.  

#### **6.4 Multi-Modal Embeddings**  
- **Fusion Method**: **Concatenation of Text + Image embeddings**  
- **Final Embedding Shape**: `[768 + 2048 = 2816]`  
- **Purpose**:
  - Provides a **unified representation** of both text and images.
  - Helps models **jointly learn** from both modalities.  

#### **6.5 Storage and Diagnostics**  
- The embeddings are saved in `embeddings_final.pt` for efficient retrieval.  
- **Emergency Diagnostic Check**:
  - Ensures no missing (`NaN`) or infinite values in embeddings.
  - Checks **mean and standard deviation** to identify anomalies.  

This embeddings-based approach strengthens the **multi-modal hate speech detection system**, improving its ability to understand complex relationships between text and images.



---

# Dataset:

- Creating a hate speech detection system using deep learning requires datasets encompassing a wide range of text, audio, and images (if applicable) to understand and detect hate speech across multiple modalities. Here are some common datasets often used for multi-modal hate speech detection:

**MMHS150K (Multi-Modal Hate Speech 150K)**:

- Contains 150,000 tweets, each with an associated image and labels for offensive and hate speech.
- Focused on Twitter data, this dataset captures real-world examples of hate speech in a multi-modal format.
- [Dataset link](https://github.com/FirojIslam/Multimodal-Hate-Speech)

This dataset is accompanied by pre-processing steps, especially for dealing with images and text separately, before feeding them into a deep learning model such as a BERT+ResNet setup for multi-modal classification tasks.

---
