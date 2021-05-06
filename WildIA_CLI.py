import fire
import pickle 
import logging
import pandas
import PySimpleGUI as sg
from pathlib import Path
import os
import WildIA_AI as testModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Classifier:
    """
    Train or predict from dataset
    """
    def predict(self):

        layout = [
                 [sg.Text('Choose your input image folder and output path')],
                 [sg.InputText('Image folder'), sg.FolderBrowse()],
                 [sg.Text('Output file name'), sg.InputText()],
                 [sg.Submit(), sg.Cancel()],
                 ]
        
        event, values = sg.Window("WildIA GUI").Layout(layout).Read()

        predict_data_path = os.path.relpath(values[0])
        output_path = './'+values[1]

        print(predict_data_path)
        print(output_path)

        model_path = './WildIA_AI.py'
        """
        Predicts `predict_data_path` data using `model_path` 
        model and saves predictions to         `output_path`
        :param predict_data_path: path to data for predictions
        :param model_path: path to trained model
        :param output_path: path to save predictions
        """
        logger.info(f"Loading data for predictions from {predict_data_path} ...")
        X = predict_data_path

        logger.info(f"Loading model from {model_path} ...")
        model = testModel

        logger.info("Running model predictions...")
        probs, predicted = model.classify(X)

        logger.info(f"Saving predictions to {output_path} ...")
        #pandas.DataFrame(y_pred).to_csv(output_path)

        

        logger.info("Successfully predicted.")

if __name__ == "__main__":
    fire.Fire(Classifier)
