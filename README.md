# 🌟 Yelp Review Sentiment Classifier 

Welcome to my NLP Sentiment Analysis Project! This project classifies Yelp reviews into 1-star (negative) or 5-star (positive) categories using text content.

## 📌 Project Overview

### 📋 Objective
- Classify Yelp reviews into binary sentiment categories (1-star vs 5-star)
- Compare NLP preprocessing techniques and ML models
- Implement an efficient classification pipeline

### 🗄️ Dataset
The [Yelp Review Dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset) contains:
- `stars`: Rating (1-5)
- `text`: Review content
- `cool/useful/funny`: User votes

> **Note**: Only 1-star and 5-star reviews were used for binary classification

## 📊 Exploratory Data Analysis

### Review Length Distribution
![BoxPlot](BoxPlot.png)

*Distribution of text length across different star ratings*

### Rating Distribution
![CountPlot](CountPlot.png)

*Count of reviews per star rating*

### Feature Correlations
![HeatMap](HeatMap.png)

*Correlation matrix between different features*

### Text Length vs Rating
![GMap with HistPlot](<GMap with HistPlot.png>)
*Relationship between text length and star ratings*

## 🛠 Tech Stack
| Category        | Tools                          |
|-----------------|--------------------------------|
| Core Language   | Python 3.x                     |
| NLP Processing  | NLTK, SpaCy                    |
| ML Framework    | Scikit-Learn                   |
| Data Handling   | Pandas, NumPy                  |
| Visualization   | Matplotlib, Seaborn            |

## 🔍 Implementation

### 2️⃣ Model Performance
**Without Text Processing**

<img width="458" height="218" alt="C-R   C-M" src="https://github.com/user-attachments/assets/b7138cd5-1b1d-4241-9cb9-f78f367b71fd" />


Confusion matrix and classification report

**With Text Processing**

<img width="447" height="171" alt="C-R   C-M using Text Processing" src="https://github.com/user-attachments/assets/a3a21efe-b6dc-4449-8af9-15e4124e2549" />


Improved results after text processing

## #🚀 Getting Started
### Prerequisites

Python 3.8+
VS Code with Python extension
Jupyter Notebook (optional)

### Installation

git clone https://github.com/safikasi/NLP-Project.git

cd NLP-Project

pip install -r requirements.txt

## 🌐 Connect With Me
**Safwan Khan Kasi**  
DS & ML Enthusiast   

[![GitHub](https://img.shields.io/badge/GitHub-safikasi-blue?logo=github)](https://github.com/safikasi)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Safwan_Kasi-blue?logo=linkedin)](https://www.linkedin.com/in/safwan-kasi-2b5358292/)
