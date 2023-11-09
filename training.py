from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import logging
from configs import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CFG = Config(production=False).config


###################Load config.json and get path variables

final_data_path = CFG["final_data_path"]
model_path = os.path.join(CFG['output_model_path']) 
output_model_train_path = os.path.join(
    os.getcwd(),
    model_path,
    'trainedmodel.pkl'
)


#################Function for training the model
def train_model(final_data_path, output_model_train_path):
    # Training with logistic regression 

    logger.info('Training model....')
    #use this logistic regression for training
    lr_model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #Prepare data
    final_df = pd.read_csv(final_data_path)
    final_df = final_df.drop(["corporation"], axis=1)
    X = final_df.iloc[:, :-1]
    y = final_df.iloc[:, -1]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=0)
    
    #fit the logistic regression to your data
    lr_model.fit(X_train, y_train)

    #write the trained model to your workspace in a file called trainedmodel.pkl
    with open(output_model_train_path, 'wb') as f:
        pickle.dump(lr_model, f)
    logger.info('Model saved at {}'.format(output_model_train_path))
    logger.info('Training model finished')

if __name__ == '__main__':
    train_model(
        final_data_path = final_data_path, 
        output_model_train_path = output_model_train_path
        )
