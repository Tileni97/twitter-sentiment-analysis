# Twitter Sentiment Analysis using Natural Language Processing

## Overview
This project implements a machine learning model for sentiment analysis on Twitter data using Natural Language Processing (NLP) techniques. The model classifies tweets as either positive or negative, achieving 78% accuracy on test data.

## Features
- Text preprocessing including tokenization, stemming, and stopword removal
- TF-IDF vectorization for feature extraction
- Logistic Regression model for sentiment classification
- Model evaluation with confusion matrix and classification metrics
- Trained model persistence using pickle

## Dataset
The project uses the Sentiment140 dataset containing 1.6 million tweets labeled with sentiment (positive/negative). The dataset includes:
- target: Sentiment label (0 = negative, 1 = positive)
- id: Tweet ID
- date: Tweet timestamp
- flag: Query flag
- user: User who tweeted
- text: Tweet content

## Requirements
- Python 3.x
- pandas
- numpy
- nltk
- scikit-learn
- matplotlib
- seaborn
- kaggle

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/twitter-sentiment-analysis.git

# Install required packages
pip install -r requirements.txt

# Download NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"
```

## Usage
1. Set up your Kaggle API credentials in `~/.kaggle/kaggle.json`
2. Run the Jupyter notebook: `Twitter_Sentimental_Analysis_using_ML.ipynb`
3. The trained model will be saved as 'trained_model.sav'

## Model Performance
- Training Accuracy: 81%
- Test Accuracy: 78%
- Balanced performance across positive and negative sentiments
- Detailed metrics available in confusion matrix and classification report

## Project Structure
```
twitter-sentiment-analysis/
├── Twitter_Sentimental_Analysis_using_ML.ipynb
├── trained_model.sav
├── requirements.txt
└── README.md
```

## Future Improvements
- Experiment with different ML algorithms (Random Forest, Naive Bayes)
- Implement deep learning approaches
- Add real-time tweet analysis capability
- Enhance text preprocessing pipeline
- Add cross-validation

## License
MIT License - feel free to use this code for your own projects

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.