# VibeVerify: Music Genre Classification and Sound Effect Recognition

## Project Overview

**VibeVerify** is an AI-powered system designed to classify music genres and recognize sound effects from audio clips. The project uses machine learning models such as Random Forest, SVM, KNN, Logistic Regression, and Decision Tree to classify music genres based on pre-extracted audio features like MFCCs, spectral centroid, chroma features, and more.

## Dataset

The dataset contains pre-extracted features from audio files across 10 different music genres:
- **Genres**: Blues, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock.
- **Features**: 
  - Chroma features (`chroma_stft_mean`, `chroma_stft_var`)
  - RMS energy (`rms_mean`, `rms_var`)
  - Spectral features (`spectral_centroid_mean`, `spectral_bandwidth_mean`)
  - MFCCs (`mfcc1_mean` to `mfcc20_mean`)
  
The dataset is stored in a CSV file (`dataset.csv`) that contains these features along with the corresponding genre label for each audio file.

## Code Structure

- **data_preprocessing.ipynb**: This notebook handles data loading from the CSV file, cleaning (e.g., dropping unnecessary columns), normalizing the features using `StandardScaler()`, and splitting the data into training and testing sets.
- **model_training.ipynb**: This notebook trains multiple machine learning models (Random Forest, SVM, KNN, Logistic Regression, Decision Tree) on the pre-processed data and evaluates their performance.
- **images/**: This folder contains optional spectrogram images (if you choose to include them).
- **models/**: This folder can store saved models (e.g., trained Random Forest model).

## Models Trained

The following machine learning models were trained on the dataset:

1. **Random Forest Classifier**
2. **Support Vector Machine (SVM)**
3. **K-Nearest Neighbors (KNN)**
4. **Logistic Regression**
5. **Decision Tree**

### Model Performance

| Model                | Accuracy (%) |
|----------------------|--------------|
| Random Forest         | 65.50        |
| Support Vector Machine| 70.00        |
| K-Nearest Neighbors   | 65.50        |
| Logistic Regression   | 69.00        |
| Decision Tree         | 46.50        |

Based on these results, the **Support Vector Machine (SVM)** model performed the best with an accuracy of **70%**, followed by Logistic Regression with an accuracy of **69%**.

## How to Run the Code

### Prerequisites

Make sure you have Python installed along with the following libraries:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install all dependencies using the `requirements.txt` file:

bash
pip install -r requirements.txt

Steps to Run:
Clone this repository:
bash
git clone https://github.com/yourusername/VibeVerify-Project.git

Navigate to the project directory:
bash
cd VibeVerify-Project

Install dependencies:
bash
pip install -r requirements.txt

Run the Jupyter notebooks:
Open data_preprocessing.ipynb to preprocess and visualize the data.
Open model_training.ipynb to train multiple machine learning models and evaluate their performance.
Optional: Spectrogram Images
If you want to include spectrogram images for CNN-based models or visualization purposes, you can place them in the images/ folder.
Data Visualization
The following visualizations are included in the notebooks:
Box plots showing how MFCCs and spectral features are distributed across different genres.
Count plot showing the distribution of genres in the dataset.
