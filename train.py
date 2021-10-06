import numpy as np
import pandas as pd 
import joblib 
from sklearn.linear_model import LinearRegression 
dataset = pd.read_csv('hiring.csv')
print(dataset)
dataset.experience.fillna(0,inplace = True)   # replacing null value which available in experience column with 0 .
dataset.test_score.fillna(dataset.test_score.mean(),inplace = True)  ## replacing null value which is available in test_score with mean value
# Separate independent and dependent variable 

X = dataset.iloc[:,:3]
def convert_to_int(word):
    word_dict = {'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine' : 9 ,'ten':10,'eleven':11,0:0}
    return word_dict[word]

X['experience'] = X.experience.apply(lambda x : convert_to_int(x) )
print(X)
y= dataset.iloc[:,-1]   # Its a regression problem 

regression = LinearRegression()
regression.fit(X,y)
print('Model training is done')
print(regression.predict([[1,8,9]]))  # Lets predict how much salary i will get ig a person have experience =1,test_score = 8,interview_score =9
joblib.dump(regression,'hiring_model.pkl')        # For saving the model 

