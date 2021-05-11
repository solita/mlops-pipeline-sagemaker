import time
import logging
from pathlib import Path
import boto3

logging.getLogger().setLevel(logging.INFO)
DATA_PATH = Path("/opt/ml/processing").resolve()


def upload_mlmodel(mlfilename):
    """Upload the .mlmodel artifact to API-enabled S3 bucket"""

    model_file = str(DATA_PATH/"mlmodel/bitti.mlmodel")

    bucket_name = "magazine-monitor"
    s3 = boto3.resource('s3')
    s3.Bucket(bucket_name).upload_file(model_file, f"Models/{mlfilename}")
    logging.info("Uploaded %s to the API-enabled S3 bucket.", mlfilename)


def upload_model_card(jsonfilename):
    """Upload the model card to a public S3 bucket w. static hosting"""

    model_card_html_file = str(DATA_PATH/"model_card/model_card.html")
    model_card_json_file = str(DATA_PATH/"model_card/model_card.json")
    output_file_name = "model_card_latest.html"
    bucket_name = "magazine-monitor"
    bucket_name_static_page = "aigamodelcards.com"
    s3 = boto3.resource('s3')
    s3.Bucket(bucket_name_static_page).upload_file(
            Filename=model_card_html_file,
            Key=output_file_name,
            ExtraArgs={'ContentType': "text/html"})
    s3.Bucket(bucket_name).upload_file(
            Filename=model_card_json_file,
            Key=f"Models/{jsonfilename}")
    logging.info("Uploaded %s to %s, and %s to %s", output_file_name,
                 bucket_name_static_page, jsonfilename, bucket_name)


if __name__ == "__main__":
    datestring = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    upload_mlmodel(mlfilename=f'bitti-{datestring}.mlmodel')
    upload_model_card(jsonfilename=f'bitti-{datestring}.json')
