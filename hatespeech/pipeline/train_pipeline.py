import os
import sys
from hatespeech.logger import logging
from hatespeech.exception import CustomException
from hatespeech.components.data_ingestion import DataIngestion
from hatespeech.components.data_transformation import DataTransformation
from hatespeech.components.model_trainer import ModelTrainer
from hatespeech.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)
from hatespeech.entity.artifact_entity import (
    DataIngestionArtifacts,
    DataTransformationArtifacts,
    ModelTrainerArtifacts,
)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()

    def start_data_ingestion(self) -> DataIngestionArtifacts:
        """
        Initiates the data ingestion process.
        
        Returns:
            DataIngestionArtifacts: The artifacts resulting from data ingestion.
        """
        logging.info("Entered the start_data_ingestion method of TrainPipeline class")
        try:
            logging.info("Getting the data from GCloud Storage bucket")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)

            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train and valid from GCloud Storage")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifacts

        except Exception as e:
            logging.error("Error during data ingestion: %s", str(e))
            raise CustomException(e, sys) from e

    def start_data_transformation(self, data_ingestion_artifacts: DataIngestionArtifacts) -> DataTransformationArtifacts:
        """
        Initiates the data transformation process.
        
        Parameters:
            data_ingestion_artifacts (DataIngestionArtifacts): The artifacts obtained from data ingestion.
        
        Returns:
            DataTransformationArtifacts: The artifacts resulting from data transformation.
        """
        logging.info("Entered the start_data_transformation method of TrainPipeline class")
        try:
            data_transformation = DataTransformation(
                data_ingestion_artifacts=data_ingestion_artifacts,
                data_transformation_config=self.data_transformation_config
            )

            data_transformation_artifacts = data_transformation.initiate_data_transformation()
            
            logging.info("Exited the start_data_transformation method of TrainPipeline class")
            return data_transformation_artifacts

        except Exception as e:
            logging.error("Error during data transformation: %s", str(e))
            raise CustomException(e, sys) from e

    def start_model_trainer(self, data_transformation_artifacts: DataTransformationArtifacts) -> ModelTrainerArtifacts:
        """
        Initiates the model training process.
        
        Parameters:
            data_transformation_artifacts (DataTransformationArtifacts): The artifacts obtained from data transformation.
        
        Returns:
            ModelTrainerArtifacts: The artifacts resulting from model training.
        """
        logging.info("Entered the start_model_trainer method of TrainPipeline class")
        try:
            model_trainer = ModelTrainer(
                data_transformation_artifacts=data_transformation_artifacts,
                model_trainer_config=self.model_trainer_config
            )
            model_trainer_artifacts = model_trainer.initiate_model_trainer()
            logging.info("Exited the start_model_trainer method of TrainPipeline class")
            return model_trainer_artifacts

        except Exception as e:
            logging.error("Error during model training: %s", str(e))
            raise CustomException(e, sys) from e

    def run_pipeline(self) -> None:
        """
        Orchestrates the data ingestion, transformation, and model training process.
        """
        logging.info("Entered the run_pipeline method of TrainPipeline class")
        try:
            data_ingestion_artifacts = self.start_data_ingestion()
            data_transformation_artifacts = self.start_data_transformation(
                data_ingestion_artifacts=data_ingestion_artifacts
            )

            model_trainer_artifacts = self.start_model_trainer(
                data_transformation_artifacts=data_transformation_artifacts
            )

            logging.info("Pipeline completed successfully.")  # Success log
            logging.info("Exited the run_pipeline method of TrainPipeline class")   

        except Exception as e:
            logging.error("Error in pipeline execution: %s", str(e))
            raise CustomException(e, sys) from e
