"""Endpoint definition for AIGA model retriever"""

import base64
import json
import boto3

S3_CONNECTOR = boto3.client('s3')
S3_BUCKET = 'magazine-monitor'
S3_PREFIX = "Models"
URL_PRESIGNED = True
URL_EXPIRES_IN = 60


def list_mlmodels():
    """Returns a list of stored models in CoreML format"""

    file_list = S3_CONNECTOR.list_objects_v2(Bucket=S3_BUCKET,
                                             Prefix=S3_PREFIX)
    res = ['/'.join(c['Key'].split('/')[1:]) for c in file_list['Contents']]

    return [r for r in res if ".mlmodel" in r]


def get_mlmodel(key):
    """Returns base64-encoded .mlmodel binary file from S3 storage

    FIXME: this doesn't work for payload sizes of 6 MB and larger, because the
           hard limits of AWS Lambda are hit.
    """

    response = S3_CONNECTOR.get_object(Bucket=S3_BUCKET,
                                       Key=f"{S3_PREFIX}/{key}")

    return base64.b64encode(response['Body'].read()).decode('utf-8')



def get_presigned_url(key, expires_in=URL_EXPIRES_IN):
    """Returns presigned url with a self-destruct ticker"""

    presigned_url = S3_CONNECTOR.generate_presigned_url(
        ClientMethod='get_object',
        Params={'Bucket': S3_BUCKET, 'Key': f"{S3_PREFIX}/{key}"},
        ExpiresIn=URL_EXPIRES_IN)

    return presigned_url


def lambda_handler(event, context):
    """GET endpoint for the model retriever"""

    try:
        key = event["queryStringParameters"]['key']
    except (KeyError, TypeError) as _:
        key = None

    if key and not URL_PRESIGNED:
        # return a model binary with octet-stream mimetype according to
        # RFC2046 (https://www.rfc-editor.org/rfc/rfc2046.txt)
        return {
            'headers': {"Content-Type": "application/octet-stream"},
            'statusCode': 200,
            'body': get_mlmodel(key),
            'isBase64Encoded': True
        }

    if key:
        # generate and return a presigned URL for downloading the S3 file
        return {
            'headers': {"Content-Type": "application/json"},
            'statusCode': 200,
            'body': get_presigned_url(key)
        }

    # if no particular file was requested, list all models in a JSON array
    return {
        'headers': {"Content-type": "application/json"},
        'statusCode': 200,
        'body': json.dumps(list_mlmodels()),
    }
