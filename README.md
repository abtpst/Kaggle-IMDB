# Kaggle-IMDB

Here I am trying to solve the sentiment analysis problem for movie reviews. The problem is taken from the Kaggle competition 

https://www.kaggle.com/c/word2vec-nlp-tutorial

I will be using python as my programming language. For this, I have used the Anaconda 2.7 package.

I have used three different classifiers to solve this problem. All of the classifiers have a common pre processing step where I perform data cleanup and then use TfidfVectorizer for feature selection

**Instructions to run**

1. Clone this git repo to a suitable location on your machine.

2. Download and unzip the following data files

	testData.tsv 	
	labeledTrainData.tsv
	
	from https://www.kaggle.com/c/word2vec-nlp-tutorial/data and store them in the `data` folder

3. Run the `classify.py` script in the `imdbMain` package. This will make predictions as per all three algorithms.

4. Once the script has terminated, the final predictions should be in the `results` folder
