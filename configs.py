import os
import json

class Config:
    def __init__(self, production=False):
        self.production = production
        self.load_config()
        self.create_folders()

    def load_config(self):
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        if self.production:
            self.config['input_folder_path'] = "sourcedata"
            self.config['output_model_path'] = "models"

        # Save result paths
        self.config["test_data_csv_path"] = os.path.join(
            os.getcwd(),
            self.config["test_data_path"],
            'testdata.csv'
        )
        self.config["final_data_path"] = os.path.join(
            os.getcwd(),
            self.config["output_folder_path"],
            'finaldata.csv'
        )
        self.config["api_returns_path"] = os.path.join(
            os.getcwd(),
            self.config["output_model_path"],
            'apireturns2.txt' if self.production else 'apireturns.txt'
        )
        self.config["cfm_path"] = os.path.join(
            os.getcwd(),
            self.config["output_model_path"],
            'confusionmatrix2.png' if self.production else 'confusionmatrix.png'
        )

    def create_folders(self):
        for folder_path in [self.config["output_model_path"], self.config["prod_deployment_path"]]:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

if __name__ == '__main__':
    config = Config()
    print(config.config)
