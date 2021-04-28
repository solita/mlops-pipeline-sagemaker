import time
import logging
import boto3

logging.getLogger().setLevel(logging.INFO)


def upload_mlmodel():
    # TODO: upload the model card to a S3 public bucket once it's ready
    model_file = "/opt/ml/processing/model/bitti.mlmodel"
    datestring = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
    mlfilename = f'bitti-{datestring}.mlmodel'
    bucket_name = "magazine-monitor"
    s3 = boto3.resource('s3')
    s3.Bucket(bucket_name).upload_file(model_file, f"Models-test/{mlfilename}")
    logging.info("Uploaded %s to the API-enabled S3 bucket.", mlfilename)


if __name__ == "__main__":
    upload_mlmodel()
