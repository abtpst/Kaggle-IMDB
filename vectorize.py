import pandas as pd
import utilities.preProc as preProc
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer as TFIV

train = pd.read_csv('../../data/labeledtrainData.tsv', header=0, delimiter="\t", quoting=3)

predCol = train['sentiment']

pickle.dump(predCol,open("../../classifier/logisticRegression/predCol","wb"))                

trainData = []

numRevs = len(train['review'])

for i in range(0,numRevs):
    
    if( (i+1)%2000 == 0 ):
            
            print ("Train Review %d of %d\n" % ( i+1, numRevs ))
            
    trainData.append(" ".join(preProc.Sentiment_to_wordlist(train['review'][i])))

test = pd.read_csv('../../data/testData.tsv', header=0, delimiter="\t", quoting=3 )

testdata = []

numRevs = len(test['review'])

for i in range(0,numRevs):
    
    if( (i+1)%2000 == 0 ):
            
            print ("Test Review %d of %d\n" % ( i+1, numRevs ))
            
    testdata.append(" ".join(preProc.Sentiment_to_wordlist(test['review'][i])))

print("Defining TFIDF Vectorizer")        

tfIdfVec = TFIV(
                    min_df=3, # When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold
                    max_features=100000, # If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
                    strip_accents='unicode', # Remove accents during the preprocessing step. 'ascii' is a fast method that only works on characters that have an direct ASCII mapping.
                                             # 'unicode' is a slightly slower method that works on any characters.
                    analyzer='word', # Whether the feature should be made of word or character n-grams.. Can be callable.
                    token_pattern=r'\w{1,}', # Regular expression denoting what constitutes a "token", only used if analyzer == 'word'. 
                    ngram_range=(1,5), # The lower and upper boundary of the range of n-values for different n-grams to be extracted.
                    use_idf=1, # Enable inverse-document-frequency reweighting.
                    smooth_idf=1, # Smooth idf weights by adding one to document frequencies.
                    sublinear_tf=1, # Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
                    stop_words = 'english' # 'english' is currently the only supported string value.
                )

pickle.dump(tfIdfVec,open("../../classifier/logisticRegression/100000Mxtfiv15gram","wb"))

combineData = trainData + testdata # Combine both to fit the TFIDF vectorization.

trainLen = len(trainData)

print("Fitting")

tfIdfVec.fit(combineData) # Learn vocabulary and idf from training set.

print("Transforming")

combineData = tfIdfVec.transform(combineData) # Transform documents to document-term matrix. Uses the vocabulary and document frequencies (df) learned by fit (or fit_transform).
pickle.dump(combineData,open("../../classifier/logisticRegression/100000MxcombineData15","wb"))   
print("Fitting and transforming done")

trainAfterFit = combineData[:trainLen] # Separate back into training and test sets. 
pickle.dump(trainAfterFit,open("../../classifier/logisticRegression/100000MxtrainAfterFit15","wb"))    

testAfterFit = combineData[trainLen:]
pickle.dump(testAfterFit,open("../../classifier/logisticRegression/100000MxtestAfterFit15","wb"))
