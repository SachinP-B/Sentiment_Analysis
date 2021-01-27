#!/usr/bin/env python
# coding: utf-8


# Importing Libraries
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from Naive_Bayes_functions import *

#########################################################################################################################################
# # Dataset creating and level 1 preprocessing
# 
# Here we just load the data into python environment and create a pandas DataFrame with required columns.
# We also store the final outcome as CSV inorder to just load the data for model building and experiments


#Uncomment the line below to do this one time process
#If preprocessed_data_lvl_1.csv is already present no need to do it

lines = []
for the_text in open('naive_bayes_data.txt', encoding='utf8'):
    lines.append(the_text.strip('\n'))
the_data = pd.DataFrame(columns=["Id","Category","Text","Sentiment"],index=range(0, len(lines)))
for idx,line in enumerate(lines):
    tokens = line.split(" ")    
    the_data.loc[idx,"Category"] = tokens[0]
    the_data.loc[idx,"Sentiment"] = tokens[1]
    the_data.loc[idx,"Id"] = tokens[2].split(".")[0]
    the_data.loc[idx,"Text"] = ' '.join(word for word in tokens[3:])
    
the_data.to_csv("preprocessed_data_lvl_1.csv", index = False)

################################################################################################################################################

# # Text classification Preprocessing
# 
# Here we do the the usual preprocessing required for text classification. This includes
# 1. Tokenisation
# 2. Punctuation removal (if required)
# 3. Stop words removal (if required)
# 4. Transforming text into features for building any Classification model (X)
# 
# Note: Tokenisation could have been done in level 1 prepocessing itself but to make the process in well strucuted manner for any given CSV file with a text columns, we are doing it here


#loading the preprocessed data
the_data = pd.read_csv("preprocessed_data_lvl_1.csv")


#Tokenising
the_data["tokenised_text"] = tokeniser(the_data["Text"])
#Uncomment to see the intermediate result
#the_data[["Text", "tokenised_text"]].head()


#Punctuation removal
the_data["tokenised_text"] = punctuation_remover(the_data["tokenised_text"])
#Uncomment to see the intermediate result
#the_data[["Text", "tokenised_text"]].head()


#test Train split
train_data, test_data = test_train_split(the_data, 0.8)


#Training the Model with train_data
priors = naive_bayes_training(train_data, "Sentiment")




#Testing the fitted model on the test_data
test_data = naive_bayes_prediction(test_data, priors)


#Model Performance Metrics
confusion_matrix, accuracy, precision, recall, f1_score = model_performance(test_data["Sentiment"], test_data["prediction"])
print("Confusion Matrix")
print(confusion_matrix)
print("\n\n")
print("Accuracy:", accuracy, "%")
print("Precision:", precision, "%")
print("Recall:", recall, "%")
print("f1-score:", f1_score, "%")

test_data.to_csv("Final_prediction.tsv", sep= "\t", encoding='utf-8', index = False )




