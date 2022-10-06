# EgyptionCompaniesReviews_Sentiment_analysis

### This app was made by [Fahd Seddik](https://github.com/FahdSeddik) and developed by our data science team.

This is a Streamlit app for sentiment analysis that we uses two models.

1- XLM RoBERTa Pre-Trained model on Hugging Face for English Sentiment analysis. 

2- Arabic Sentiment Analysis model that our team built it from scratch:

      • For a different corporation, we scraped a 67k Arabic reviews. [Facebook, Instagram, Google Play, App Store, Google Maps, and Glassdoor].

      •  Data preprocessing, EDA and trained a various machine learning models.

      • We scrapes Twitter for tweets about a specific company. The tweets are then fed into a model for sentiment analysis.

# Interface
Below is a video demo of the app.   

https://user-images.githubusercontent.com/60902991/194188937-ff6f1283-c206-403f-b4ed-f3357979adb3.mp4

 # Installation
To be able to use this app, please follow the instructions below. First, you need to install requirements using the following command.
```bash
pip install -r requirements.txt
```
After that, you need to download this pre-trained model from [Hugging Face](https://huggingface.co/akhooli/xlm-r-large-arabic-sent). 
```python
from transformers import pipeline
import tokenizers
# this will download 2 GB
nlp = pipeline("sentiment-analysis", model='akhooli/xlm-r-large-arabic-sent')
# Save it in the same app folder
# .save_pretrained(path)
# 'XLM-R-L-ARABIC-SENT' is the folder name of the model
nlp.save_pretrained('XLM-R-L-ARABIC-SENT')
```
This will produce a folder that has the model. ***Please include the folder in the same directory as 'app.py'.***  
In case you want to replace this model with another, you want to download your model and edit the `setup_model()` function. Implementation is shown below.
```python
def setup_model():
    """
    Setup Model
    """
    #*************************************************
    #  -==EDIT THE LINE BELOW WITH YOUR OWN MODEL==-
    #*************************************************
    nlp = pipeline("sentiment-analysis", model='XLM-R-L-ARABIC-SENT')
    return nlp
```
Now to run the app, just simply run the command below in a terminal.
```bash
streamlit run app.py
```
# About 
This project was made by the data science team during our internship at Digital Factory-Banque Misr .
