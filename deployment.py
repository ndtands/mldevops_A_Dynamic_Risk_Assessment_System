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
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CFG = Config(production=False).config



##################Load config.json and correct path variable
prod_deployment_path = CFG['prod_deployment_path']
output_model_path = CFG['output_model_path']
output_folder_path = CFG['output_folder_path']

####################function for deployment
def store_model_into_pickle(output_model_path, output_folder_path, prod_deployment_path):
    """
    Store the model into pickle file and save it in the deployment folder
    """
    lr_model_path = os.path.join(os.getcwd(), output_model_path, 'trainedmodel.pkl')
    prod_deployment_path = os.path.join(os.getcwd(), prod_deployment_path)
    latestscore_path = os.path.join(os.getcwd(), output_model_path, 'latestscore.txt')
    ingestfiles_path = os.path.join(os.getcwd(), output_folder_path, 'ingestedfiles.txt')

    shutil.copy(lr_model_path, prod_deployment_path)
    shutil.copy(latestscore_path, prod_deployment_path)
    shutil.copy(ingestfiles_path, prod_deployment_path)

    # Sử dụng logging để ghi lại thông tin
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info('All files saved at {}'.format(prod_deployment_path))
        

if __name__ == '__main__':
    store_model_into_pickle(
        output_model_path = output_model_path,
        output_folder_path = output_folder_path,
        prod_deployment_path = prod_deployment_path
        )
