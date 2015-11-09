from sklearn.linear_model import SGDClassifier as SGD
from sklearn.grid_search import GridSearchCV

import pickle
import pandas as pd

trainAfterFit = pickle.load(open("../../classifier/logisticRegression/100000MxtrainAfterFit15","rb"))

predCol = pickle.load(open("../../classifier/logisticRegression/predCol","rb"))

testAfterFit = pickle.load(open("../../classifier/logisticRegression/100000MxtestAfterFit15","rb")) 

test = pd.read_csv('../../data/testData.tsv', header=0, delimiter="\t", quoting=3 )

sgd_params = {'alpha': [0.00006, 0.00007, 0.00008, 0.0001, 0.0005]} # Constant that multiplies the regularization term. Defaults to 0.0001

model_SGD = GridSearchCV(
                         SGD(
                             random_state = 0, # The seed of the pseudo random number generator to use when shuffling the data.
                             shuffle = True, # Whether or not the training data should be shuffled after each epoch. Defaults to True.
                             loss = 'modified_huber'
                             
                             # The loss function to be used. Defaults to 'hinge', which gives a linear SVM. 
                             # The 'log' loss gives logistic regression, a probabilistic classifier. 
                             # 'modified_huber' is another smooth loss that brings tolerance to outliers as well as probability estimates. 
                             # 'squared_hinge' is like hinge but is quadratically penalized. 'perceptron' is the linear loss used by the perceptron algorithm. 
                             # The other losses are designed for regression but can be useful in classification as well; see SGDRegressor for a description.
                            
                             ), 
                         sgd_params,
                         scoring = 'roc_auc', # A string (see model evaluation documentation) or a scorer callable object / function with signature scorer(estimator, X, y).
                         cv = 20 # If an integer is passed, it is the number of folds.
                        ) 

model_SGD.fit(trainAfterFit,predCol) # Fit the model.

print(model_SGD.grid_scores_)

SGD_result = model_SGD.predict_proba(testAfterFit)[:,1]
SGD_output = pd.DataFrame(data={"id":test["id"], "sentiment":SGD_result})
SGD_output.to_csv('../../submits/100000Mx15GramSGDwithNumsAndSmileys.csv', index = False, quoting = 3)
