import os
import re
import sys
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from hatespeech.logger import logging
from hatespeech.exception import CustomException
from hatespeech.entity.config_entity import DataTransformationConfig
from hatespeech.entity.artifact_entity import DataIngestionArtifacts,DataTransformationArtifacts
from nltk.corpus import stopwords
from hatespeech.exception import CustomException


class DataTransformation:
    def __init__(self,data_transformation_config: DataTransformationConfig,data_ingestion_artifacts:DataIngestionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts

    
    def imbalance_data_cleaning(self):

        try:
            logging.info("Entered into the imbalance_data_cleaning function")
            imbalance_data=pd.read_csv(self.data_ingestion_artifacts.imbalance_data_file_path)
            imbalance_data.drop(self.data_transformation_config.ID,axis=self.data_transformation_config.AXIS , 
            inplace = self.data_transformation_config.INPLACE)
            logging.info(f"Exited the imbalance data_cleaning function and returned imbalance data {imbalance_data}")
            return imbalance_data 
        except Exception as e:
            raise CustomException(e,sys) from e 

    
    def raw_data_cleaning(self):
        
        try:
            logging.info("Entered into the raw_data_cleaning function")
            raw_data = pd.read_csv(self.data_ingestion_artifacts.raw_data_file_path)
            raw_data.drop(self.data_transformation_config.DROP_COLUMNS,axis = self.data_transformation_config.AXIS,
            inplace = self.data_transformation_config.INPLACE)

            raw_data[raw_data[self.data_transformation_config.CLASS]==0][self.data_transformation_config.CLASS]=1
            
            # replace the value of 0 to 1
            raw_data[self.data_transformation_config.CLASS].replace({0:1},inplace=True)

            # Let's replace the value of 2 to 0.
            raw_data[self.data_transformation_config.CLASS].replace({2:0}, inplace = True)

            # Let's change the name of the 'class' to label
            raw_data.rename(columns={self.data_transformation_config.CLASS:self.data_transformation_config.LABEL},inplace =True)
            logging.info(f"Exited the raw_data_cleaning function and returned the raw_data {raw_data}")
            return raw_data

        except Exception as e:
            raise CustomException(e,sys) from e


    
    def concat_dataframe(self):

        try:
            logging.info("Entered into the concat_dataframe function")
            # Let's concatinate both the data into a single data frame.
            frame = [self.raw_data_cleaning(), self.imbalance_data_cleaning()]
            df = pd.concat(frame)
            print(df.head())
            logging.info(f"returned the concatinated dataframe {df}")
            return df

        except Exception as e:
            raise CustomException(e, sys) from e


    

    def concat_data_cleaning(self, words):
        try:
            logging.info("Entered into the concat_data_cleaning function")
            
            # Initialize stemmer and stopword set
            stemmer = nltk.SnowballStemmer("english")
            stopword = set(stopwords.words('english'))
            
            # Lowercase the text
            words = str(words).lower()
            
            # Remove unwanted patterns (e.g., URLs, HTML tags, punctuation, numbers)
            words = re.sub(r'\[.*?\]', '', words)
            words = re.sub(r'https?://\S+|www\.\S+', '', words)
            words = re.sub(r'<.*?>+', '', words)
            words = re.sub(r'[%s]' % re.escape(string.punctuation), '', words)
            words = re.sub(r'\n', '', words)
            words = re.sub(r'\w*\d\w*', '', words)
            
            # Remove stopwords
            words = [word for word in words.split() if word not in stopword]
            
            # Apply stemming
            words = [stemmer.stem(word) for word in words]
            
            # Join the cleaned and stemmed words back into a single string
            words = " ".join(words)
            
            logging.info("Exited the concat_data_cleaning function")
            return words

        except Exception as e:
            logging.error(f"Error occurred in concat_data_cleaning: {str(e)}")
            raise CustomException(e, sys) from e


    
    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        try:
            logging.info("Entered the initiate_data_transformation method of Data transformation class")
            self.imbalance_data_cleaning()
            self.raw_data_cleaning()
            df = self.concat_dataframe()
            df[self.data_transformation_config.TWEET]=df[self.data_transformation_config.TWEET].apply(self.concat_data_cleaning)

            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, exist_ok=True)
            df.to_csv(self.data_transformation_config.TRANSFORMED_FILE_PATH,index=False,header=True)

            data_transformation_artifact = DataTransformationArtifacts(
                transformed_data_path = self.data_transformation_config.TRANSFORMED_FILE_PATH
            )
            logging.info("returning the DataTransformationArtifacts")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e

    



