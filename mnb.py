from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.cross_validation import cross_val_score

import numpy as np
import pickle
import pandas as pd

trainAfterFit = pickle.load(open("../../classifier/logisticRegression/100000MxtrainAfterFit15","rb"))

predCol = pickle.load(open("../../classifier/logisticRegression/predCol","rb"))

testAfterFit = pickle.load(open("../../classifier/logisticRegression/100000MxtestAfterFit15","rb")) 

test = pd.read_csv('../../data/testData.tsv', header=0, delimiter="\t", quoting=3 )

model_NB = MNB()

model_NB.fit(trainAfterFit,predCol)

print ("20 Fold CV Score for Multinomial Naive Bayes: ", np.mean(cross_val_score(model_NB, trainAfterFit, predCol, cv=20, scoring='roc_auc')))
# This will give us a 20-fold cross validation score that looks at ROC_AUC so we can compare with Logistic Regression. 

MNB_result = model_NB.predict_proba(testAfterFit)[:,1]
MNB_output = pd.DataFrame(data={"id":test["id"], "sentiment":MNB_result})
MNB_output.to_csv('../../submits/100000Mx15GramMNBwithNumsAndSmileys.csv', index = False, quoting = 3)
