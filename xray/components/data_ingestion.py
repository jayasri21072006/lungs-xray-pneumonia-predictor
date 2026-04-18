import os
import sys

from xray.entity.artifacts_entity import DataIngestionArtifact
from xray.entity.config_entity import DataIngestionConfig
from xray.exception import XRayException
from xray.logger import logging


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def get_local_data(self):
        try:
            logging.info("Using local dataset")

            data_path = self.data_ingestion_config.data_path

            if not os.path.exists(data_path):
                raise Exception(f"Path not found: {data_path}")

            logging.info("Local dataset found successfully")

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_data_ingestion(self):
        try:
            logging.info("Started data ingestion")

            self.get_local_data()

            artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_data_path,
                test_file_path=self.data_ingestion_config.test_data_path
            )

            logging.info("Completed data ingestion")

            return artifact

        except Exception as e:
            raise XRayException(e, sys)