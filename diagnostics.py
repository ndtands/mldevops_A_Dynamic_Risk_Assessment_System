
import pandas as pd
import numpy as np
import timeit
import os
import json
import logging
from configs import Config
import shutil
import pickle
import subprocess
from ingestion import merge_multiple_dataframe
from training import train_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CFG = Config(production=False).config

##################Load config.json and get environment variables
input_folder_path = CFG['input_folder_path']
test_data_path = CFG['test_data_path']
test_data_csv_path = CFG['test_data_csv_path']
prod_deployment_path = CFG['prod_deployment_path']
output_folder_path = CFG['output_folder_path']
final_data_path = CFG['final_data_path']

##################Function to get model predictions
def model_predictions(test_data_csv_path, prod_deployment_path):
    #read the deployed model and a test dataset, calculate predictions
    lr_model = pickle.load(
        open(
            os.path.join(
                os.getcwd(),
                prod_deployment_path,
                "trainedmodel.pkl"),
            'rb'))
    test_df = pd.read_csv(test_data_csv_path)
    test_df = test_df.drop(['corporation'], axis=1)

    X_test = test_df.iloc[:, :-1]
    y_pred = lr_model.predict(X_test)
    return y_pred

##################Function to get summary statistics
def dataframe_summary(test_data_csv_path):
    #calculate summary statistics here
    df = pd.read_csv(test_data_csv_path)
    df = df.drop(['corporation'], axis=1)
    X = df.iloc[:, :-1]

    # calculate summary statistics of the training data
    summary = X.agg(['mean', 'median', 'std', 'min', 'max'])
    summary_statistics = list(summary['lastmonth_activity']) + \
        list(summary['lastyear_activity']) + \
        list(summary['number_of_employees'])
    return summary_statistics

##################Function to caculate missing data
def missing_data(test_data_csv_path):
    #Calculate the percentage of missing data in the training data
    df = pd.read_csv(test_data_csv_path)
    df = df.drop(['corporation'], axis=1)

    missing_values_df = df.isna().sum() / df.shape[0]
    return missing_values_df.values.tolist()


##################Function to get timings
def execution_time(input_folder_path, prod_deployment_path):
    #calculate timing of training.py and ingestion.py

    iteration = 20
    starttime = timeit.default_timer()
    ingested_files_path = os.path.join(os.getcwd(), prod_deployment_path, 'ingestedfiles.txt')
    for i in range(iteration):
        merge_multiple_dataframe(input_folder_path, ingested_files_path, final_data_path)
    ingestion_timing = (timeit.default_timer() - starttime) / iteration
    logger.info('Ingestion time: {}'.format(ingestion_timing))

    # Calculate timing of training process
    prod_model_train_path = os.path.join(os.getcwd(), prod_deployment_path, 'trainedmodel.pkl')
    starttime = timeit.default_timer()
    for i in range(iteration):
        train_model(final_data_path, prod_model_train_path)
    training_timing = (timeit.default_timer() - starttime) / iteration
    logger.info('Training time: {}'.format(training_timing))
    return ingestion_timing, training_timing

##################Function to check dependencies

def outdated_packages_list():
    #Check for outdated packages in the requirements.txt file
    df = pd.DataFrame(columns=['package_name', 'current', 'recent_available'])
    with open("requirements.txt", "r") as file:
        strings = file.readlines()
        package_names = []
        curent_versions = []
        recent = []

        for line in strings:
            package_name, cur_ver = line.strip().split('==')
            package_names.append(package_name)
            curent_versions.append(cur_ver)
            info = subprocess.check_output(
                ['python', '-m', 'pip', 'show', package_name])
            recent.append(str(info).split('\\n')[1].split()[1])

        df['package_name'] = package_names
        df['current'] = curent_versions
        df['recent_available'] = recent
    logger.info('Outdated packages: {}'.format(df.values.tolist()))
    return df.values.tolist()


if __name__ == '__main__':
    model_predictions(test_data_csv_path, prod_deployment_path)
    dataframe_summary(test_data_csv_path)
    missing_data(test_data_csv_path)
    execution_time(input_folder_path, prod_deployment_path)
    outdated_packages_list()





    
