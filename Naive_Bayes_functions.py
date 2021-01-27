#!/usr/bin/env python
# coding: utf-8


# Importing Libraries
import numpy as np
import pandas as pd

#since there is a need for some text related preprocessing, using the power of regular expression
import re
#to optimise some counting
from collections import Counter
#used for some basic math operations
import math
import operator



def tokeniser(the_text_series):
    '''
    This function will tokenise any given text.
    Parameters
    -----------
    the_text_series: Pandas series which contains the text
    
    Returns
    -------
    tokenised_list: List of Lists with strings as tokens
    '''
    tokenised_list = []
    for the_row in the_text_series:
        tokenised_list.append(the_row.lower().split(" "))
    return tokenised_list


def punctuation_remover(tokenised_text_series, signs_to_remove = ["?!#%&\.,'\"():;"]):
    '''
    This function will remove the given punctuations from the tokenised text.
    We use regular expression to ease this process
    Parameters
    -----------
    tokenised_text_series: Pandas series which contains the tokenised text
    
    signs_to_remove: String of signs inside a list with appropriate escape sequence (See default value given)
    Returns
    -------
    tokenised_text_series_no_punct_list: List of Lists with given signs removed in tokens
    '''
    tokenised_text_series_no_punct_list = []
    for the_row in tokenised_text_series:
        empty_punctuations = [re.sub(str(signs_to_remove),"",token.strip()) for token in the_row]
        tokenised_text_series_no_punct_list.append(list(filter(None, empty_punctuations)) )
    return tokenised_text_series_no_punct_list


def test_train_split(the_data,train_fraction=0.7):
    '''
    This function is to split the train and test data after all preprocessing
    
    Parameters
    ----------
    the_data : pandas Dataframe with tokenised text
    train_fraction: float, fraction of data to be used for training (default = 0.7)
    
    Returns
    -------
    train_data: pandas DataFrame
    test_data: pandas DataFrame
    '''
    #using pandas sample function and with a random state 98 to get consistent split
    train_data = the_data.sample(frac = train_fraction, random_state=98)
    #train index
    train_index = train_data.index
    #getting test index by set difference
    test_index = list(set(the_data.index)- set(train_index))
    #test data based on test index
    test_data = the_data.iloc[test_index,:]
    return train_data, test_data


def naive_bayes_training(train_data, class_variable):
    '''
    This function is used to train the naive bayes classifier
    
    Parameters
    -----------
    train_data: pandas Dataframe, training data generated
    class_variable: Sting, name of the class variable
    
    Returns
    --------
    priors: pandas DataFrame
            contains 'class', 'class_freq', 'word_freq', 'train_data_size' information for each class
    '''
    
    #getting unique classes
    uniques_classes = train_data[class_variable].unique()
    #initialising a list to store the prior information for each class
    priors = []
    
    for the_class in uniques_classes:
        #initialising a dictionary to store class information
        class_information = {}
        #segreating data relevant to the current class alone
        class_data = train_data[train_data[class_variable] == the_class]
        #Class freq is nothing but the number of rows of data for the particular class
        class_freq = class_data.shape[0]
        #Storing class information and class frequency
        class_information["class"] = the_class
        class_information["class_freq"] = class_freq
        #Converting pandas Series into list of Lists
        tokenised_list_of_lists = class_data["tokenised_text"].values.tolist()
        #Converting list of lists to single list to aid the Counter to get word counts
        word_list = [item for sublist in tokenised_list_of_lists for item in sublist]
        #Generating a dictionary with key as words and value as its frequency
        #Using Counter to optimize the code since there are more than 50k uique words
        words_freq = Counter(word_list)
        class_information['word_freq'] = words_freq
        #appending class information to prior list
        priors.append(class_information)
    #converting a list of dictionary to pandas dataframe
    priors = pd.DataFrame(priors)
    #Adding the training size information to all rows
    priors["train_data_size"] = [train_data.shape[0]]*2
    return priors


def word_feature_conditional_probability(word_feature, word_freq_details, class_freq_details):
    '''
    This is a aid function to probability_calculation to calculate the conditional proability of each word on the text
    based on the prior word frequency
    
    Parameters
    -----------
    word_feature : String, the word for which the information is required
    word_freq_details: Dict, contains information about freq of words for a particular class
    class_freq_details: float, class freq as calculated during training
    
    Returns
    --------
    Float, connditional probabilty 
    '''
    #If the word is available in the dictionary return probability else 1.0
    if word_feature in word_freq_details.keys():
        return word_freq_details[word_feature]/class_freq_details
    else:
        return 1.0


def probability_calculation(new_instance, priors):
    '''
    This is the key function calculate the posterior probability
    
    parameters
    ----------
    new_instance: list of string, tokenised words
    priors: pandas DataFrame, contains prior information attained by training Naive bayes
    
    Returns
    --------
    probability_dict : Dict, with key as class name and values as normalised probabilities
    '''
    #Getting unique class
    unique_classes = priors["class"].values
    #Initialising Dict with class as keys
    probability_dict = dict.fromkeys(unique_classes, 1)
    
    #For each class calculating the probabilities
    for the_class in unique_classes:
        #filtering only current class's priors
        class_priors = priors[priors["class"] == the_class]
        #Initialising probability suing class freq and training sise details
        probability_dict[the_class] = class_priors["class_freq"].values[0]/class_priors["train_data_size"].values[0]
        
        #Retreiving the word freq and class freq information for the current class
        #to pass as parameters to word_feature_conditional_probability
        word_freq_details = class_priors["word_freq"].values[0]
        class_freq_details = class_priors["class_freq"].values[0]

        #Calcualting the conditional probabilities for each feature (i.e words)
        for word in new_instance:
            probability_dict[the_class] =  probability_dict[the_class] * word_feature_conditional_probability(word, word_freq_details, class_freq_details)

    # Here we are normalizing the probabilites by using the fact 
    # sum of all proabilites adds to 1
    #To avoid divide by zero error this condition is checked
    denom = sum(probability_dict.values())
    if(denom != 0):
        factor=1.0/sum(probability_dict.values())
        #Rounding off to 5 digits to avoid overflow error, still some overflow persists
        probability_dict = {k: np.round(v*factor, 5) for k, v in probability_dict.items()}

    return probability_dict


def naive_bayes_prediction(test_data, priors):
    '''
    This is prediction funciton for the Naive Bayes
    
    Parameters
    ----------
    test_data: pandas DataFrame, test instance with preprocess tokenised text columns
    priors: Pandas DataFrame, contains Model training information
    
    Returns
    --------
    test_data: pandas DataFrame
                With two additional columns:- Probas (dictionary containing probabilites of each class)
                                              Predictions (Class predicted by naive bayes model)
    '''
    #Initialising list to store probabiltiy and prediction for all class
    probas = []
    prediction = []
    #Iterating through every row
    for index,the_instance in test_data.iterrows():
        #Getting tokensined test columsn
        tokenised_text = the_instance["tokenised_text"]
        #Getting normalised probability
        predict_probas = probability_calculation(tokenised_text, priors)
        probas.append(predict_probas)
        #Assigning prediction based on probabilites
        predicted = max(predict_probas.items(), key=operator.itemgetter(1))[0]
        prediction.append(predicted)
    test_data['probas'] = probas
    test_data['prediction'] = prediction
    
    return test_data


def model_performance(actual, predicted):
    '''
    This function gives performance information about the model based on the results
    '''
    #generating confusion matrix using pandas Crosstab
    #Rows - Actual, Columns - Predicted
    df_confusion = pd.crosstab(actual, predicted, rownames=['Actual'], colnames=['Predicted'], margins=True, margins_name= "Total")
    ###### Structure of confusion matrix (for more info: https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)
    #   TN -- FP
    #   FN -- TP
    # Additional 3rd columns and rows has total respectively
    
    #Accuracy: (TP+TN)/total
    accuracy = (df_confusion.iloc[0,0]+df_confusion.iloc[1,1])/df_confusion.iloc[2,2]
    
    #Precision: TP/predicted yes--sum of predicted yes(Total column)
    precision = df_confusion.iloc[1,1]/df_confusion.iloc[2,1]
    
    #Recall: TP/actual yes--sum of actual yes (Total row) -- aka True Positive Rate
    recall = df_confusion.iloc[1,1]/df_confusion.iloc[1,2]
    
    #F1 score is harmonic mean of precision and recall
    f1_score = (2*precision*recall)/(precision+recall)
    return df_confusion, np.round(accuracy*100, 2), np.round(precision*100, 2), np.round(recall*100, 2), np.round(f1_score*100, 2)

