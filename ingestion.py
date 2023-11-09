import pandas as pd
import os
import json
from pathlib import Path
import logging
from configs import Config

CFG = Config(production=False).config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = CFG['input_folder_path']
output_folder_path = CFG['output_folder_path']
final_data_path = CFG['final_data_path']
ingested_files_path = os.path.join(
    os.getcwd(),
    output_folder_path,
    'ingestedfiles.txt'
)

# #############Function for data ingestion

def merge_multiple_dataframe(input_folder_path, ingested_files_path, final_data_path):
    """
    Merge multiple csv files into one dataframe and remove duplicates

    Args:
        input_folder_path (str): path to the input folder
        ingested_files_path (str): path to the ingested files
        final_data_path (str): path to the final data

    Returns:
        merged_df (dataframe): dataframe with no duplicates
    """
    input_path = Path(input_folder_path)
    files_list = list(input_path.glob('*.csv'))
    logging.info(f'Files in the input folder: {files_list}')

    dataframes = []
    for file_path in files_list:
        df = pd.read_csv(file_path)
        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True).drop_duplicates()
    logging.info('Merged dataframe shape: {}'.format(merged_df.shape))

    # Record the ingested files
    ingested_files = [file.name for file in files_list]
    Path(ingested_files_path).write_text('\n'.join(ingested_files))

    # Write the final dataframe to a csv file
    merged_df.to_csv(final_data_path, index=False)
    
    return merged_df

if __name__ == '__main__':
    merge_multiple_dataframe(
        input_folder_path=input_folder_path,
        ingested_files_path=ingested_files_path,
        final_data_path=final_data_path)
