# =============================================================================
# importing the required libraries
# =============================================================================
#Regular Expression library to clean data
import re                                                                      

import numpy as np

#Natural Language Tool Kit for processing the data
from nltk.stem.porter import PorterStemmer                                     

#for removing stopwords
from nltk.corpus import stopwords   

#Pandas library for creating dataframes                                           
import pandas as pd                                                            

#Scikit learn Machine learning library for python
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer                    #
from sklearn.model_selection import train_test_split                           #for splitting the dataset
from sklearn.ensemble import RandomForestClassifier                            #fitting Random Forest Classification to the Training set                                  

# =============================================================================
# dataset creation
# =============================================================================
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t')              #importing dataset,tabular data

cleaned_data = []                         #empty array to append cleaned data                                                           
  
print("Order of dataset: ",dataset.shape)                   #order of dataset

print("Size of dataset: ",dataset.size)                      #size of dataset

print(dataset.head())                                  #how dataset lookslike
# =============================================================================
# data cleaning
# =============================================================================
for i in range(0,1000):                                                        #dataset contains 1000 reviws 
  
    #removing punctuation, numbers,special cgaracters
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])                        
   
    #converting all cases to lower cases
    review=review.lower()                                                       
    
    #split the review into tokens/words
    review=review.split()                                                      
    
    #creating PorterStemmer object to take main stem of each word
    ps=PorterStemmer()                                                         
        
    # loop for stemming each word
    review=[ps.stem(word) for word in review if not word
            in set(stopwords.words('english'))]    
    
    #rejoin all words to create a string
    review=' '.join(review)                                                   
    
    #append each string to create array of clean text
    cleaned_data.append(review)                                                

# =============================================================================
#training and testing
# =============================================================================

#feature extraction
cv=CountVectorizer(max_features=1500)                                           
   
#x contains cleaned data (dependent variable) 
x=cv.fit_transform(cleaned_data).toarray()                                     
    
#y contains outcomes whether review is positive(1) or negative(0)
y=dataset.iloc[:, 1].values                                                    
                                                 
#splitting the dataset into training set and testing set in the ratio 3:1(750:250)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20
                                               )             
 
#creating classifier model,n_estimators is the number of trees
model=RandomForestClassifier(n_estimators=500,criterion='entropy')             

#training the model
model.fit(x_train,y_train)                                                     

#predicting the test set outcomes
y_prediction=model.predict(x_test)                                            

print("\nOutcomes of testing: \n")

print(y_prediction,"\n")

confusion_mat=confusion_matrix(y_test,y_prediction)                                     #evaluating the classifier using confusion matrix

print("\nConfusion Matrix:\n ",confusion_mat)

print("\nClassification Report:\n",classification_report(y_test,y_prediction))

print("Accuracy score: ",accuracy_score(y_test, y_prediction))



# =============================================================================
# evaluation of the classifier performanc
# =============================================================================

"""

print() 

TP=confusion_mat[0][0]#true positive:correct prediction for a positive detection
FN=confusion_mat[0][1]#false negative:incorrect prediction for a negative detection
FP=confusion_mat[1][0]#false positive:incorrect prediction for a positive detection
TN=confusion_mat[1][1]#true negative:correct prediction for a negative detection

TPR=(TP)/(TP+FN)
TNR=(TN)/(TN+FP)
FPR=(FP)/(FP+TN)
PPV=(TP)/(TP+FP)
accuracy=(TP+TN)/(TP+TN+FP+FN)

print("Recall/TPR: " ,TPR)
print("Fallout/FPR: ",FPR)
print("Specificity/TNR: ",TNR)
print("Precision/PPV: ",PPV)
+"""
# =============================================================================
# result/conclusion
# =============================================================================

y_p=[]
for i in y_prediction:    
    y_p.append(i)
    
nop=y_p.count(1)                                   #number of positive reviews                                                                               
non=y_p.count(0)                                   #number of negative reviews

if(non>nop):    
    print("\nOverall review: Negative")
else:
    print("\nOverall review: Positive")
