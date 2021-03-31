import time
import logging
import tarfile
import boto3

logging.getLogger().setLevel(logging.INFO)


if __name__ == "__main__":
    # load the model
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    model_file = "bitti.mlmodel"

    mlfilename = f'bitti-{time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())}.mlmodel'

    bucket_name = "magazine-monitor"
    s3 = boto3.resource('s3')
    s3.Bucket(bucket_name).upload_file(model_file, f"Models/{mlfilename}")
    logging.info("Uploaded %s to the API-enabled S3 bucket.", mlfilename)
