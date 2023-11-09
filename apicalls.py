import requests
import json
import os
import logging
from configs import Config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CFG = Config(production=False).config

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"
api_returns_path = CFG['api_returns_path']


#Call each API endpoint and store the responses
logger.info('Calling the API')
response_prediction = requests.post(
    URL + '/prediction?input_data=testdata/testdata.csv').content
response_scoring = requests.get(URL + '/scoring').content
response_summarystats = requests.get(URL + '/summarystats').content
response_diagnostics = requests.get(URL + '/diagnostics').content

responses = [
    response_prediction,
    response_scoring,
    response_summarystats,
    response_diagnostics
]

# #write the responses to your workspace
with open(api_returns_path, 'w') as file:
    file.write(str(responses))

logger.info('API responses: {}'.format(responses))
