from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
from scoring import score_model
from diagnostics import (
    dataframe_summary,
    execution_time,
    missing_data,
    outdated_packages_list
)

import logging
from configs import Config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

CFG = Config(production=True).config

input_folder_path = CFG['input_folder_path']
output_folder_pathut = CFG['output_folder_path']
prod_deployment_path = CFG['prod_deployment_path']
test_data_csv_path = CFG['test_data_csv_path']
lr_model_path = os.path.join(
    os.getcwd(),
    prod_deployment_path,
    'trainedmodel.pkl')
prediction_model = pickle.load(open(lr_model_path, 'rb'))


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():   
    #call the prediction function you created in Step 3
    # try:     
        filename = request.args.get('input_data')
        test_file_path = os.path.join(os.getcwd(), filename)
        if os.path.isfile(test_file_path) is False:
            return f"{test_file_path} doesn't exist"

        df = pd.read_csv(test_file_path)
        df = df.drop(['corporation'], axis=1)
        X = df.iloc[:, :-1]

        pred = prediction_model.predict(X)
        logger.info(f"Prediction result: {pred}")
        return str(pred), 200
    
    # except Exception as e:
    #     logger.error(f"An error occurred: {str(e)}")
    #     return "An error occurred", 500
    
#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def stats():  
    #check the score of the deployed model
    # try:         
        test_data_csv = pd.read_csv(test_data_csv_path)
        f1score = score_model(lr_model_path, test_data_csv)
        return  jsonify(f1score), 200
    # except Exception as e:
    #     logger.error(f"An error occurred: {str(e)}")
    #     return "An error occurred", 500

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():        
    #check means, medians, and modes for each column
    # try: 
        statistics = dataframe_summary(test_data_csv_path)
        return jsonify(statistics), 200
    # except Exception as e:
    #     logger.error(f"An error occurred: {str(e)}")
    #     return "An error occurred", 500

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    #check timing and percent NA values
    # try: 
        timing = execution_time(input_folder_path, prod_deployment_path)
        missing = missing_data(test_data_csv_path)
        dependencies = outdated_packages_list()
        res = {
            'timing': timing,
            'missing_data': missing,
            'dependency_check': dependencies,
        }
        return jsonify(res), 200
    # except Exception as e:
    #     logger.error(f"An error occurred: {str(e)}")
    #     return "An error occurred", 500

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
