# Kaggle-IMDB

Here I am trying to solve the sentiment analysis problem for movie reviews. The problem is taken from the Kaggle competition 

https://www.kaggle.com/c/word2vec-nlp-tutorial

I will be using python as my programming language.

I have used three different classifiers to solve this problem. All of the classifiers have a common pre processing step

**Fit and Trasnform**

Before I can build my classifiers I need to format the training and test data. I will convert the data into a form that can be used by the classifier. I will use sklearn's TfidfVectorizer. The purpose of the vectorizer is to create an underlying term document matrix from the data. This allows us to define which terms/words are important for prediction. This also involves data cleanup. TfidfVectorizer can handle some of the cleanup, but i have also explicitly performed most of the cleanup before feeding the data to TfidfVectorizer.

Please go through the well documented `vectorize.py` script to understand how TfidfVectorizer is built.
