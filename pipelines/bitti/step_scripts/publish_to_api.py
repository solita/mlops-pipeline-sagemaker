import time
import logging
from pathlib import Path
import boto3

logging.getLogger().setLevel(logging.INFO)
DATA_PATH = Path("/opt/ml/processing").resolve()


def upload_mlmodel():
    """Upload the .mlmodel artifact to API-enabled S3 bucket"""

    model_file = DATA_PATH/"mlmodel/bitti.mlmodel"
    datestring = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    mlfilename = f'bitti-{datestring}.mlmodel'
    bucket_name = "magazine-monitor"
    s3 = boto3.resource('s3')
    s3.Bucket(bucket_name).upload_file(model_file, f"Models-test/{mlfilename}")
    logging.info("Uploaded %s to the API-enabled S3 bucket.", mlfilename)


def upload_model_card():
    """Upload the model card to a public S3 bucket w. static hosting"""

    model_card_file = DATA_PATH/"model_card/model_card.html"
    output_file_name = "model_card_latest.html"
    bucket_name = "aigamodelcards.com"
    s3 = boto3.resource('s3')
    s3.Bucket(bucket_name).upload_file(model_card_file,
                                       output_file_name)
    logging.info("Uploaded %s to %s", output_file_name, bucket_name)


if __name__ == "__main__":
    upload_mlmodel()
    upload_model_card()
