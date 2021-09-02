from http import HTTPStatus
import boto3
from io import StringIO
import pandas as pd


def load_dataframe_from_csv(bucket: str, key: str, **kwargs):
    """
    Load dataframe from csv in S3 bucket
    :param bucket: S3 bucket
    :param key: S3 path
    """
    s3_resource = boto3.resource("s3")
    return pd.read_csv(s3_resource.Object(bucket, key).get()['Body'], **kwargs)


def write_dataframe_to_csv(dataframe: pd.DataFrame, bucket: str, key: str, **kwargs):
    """
    Write dataframe to csv in S3 bucket
    :param bucket: S3 bucket
    :param key: S3 path
    """
    csv_buffer = StringIO()
    dataframe.to_csv(csv_buffer, **kwargs)
    s3_resource = boto3.resource("s3")
    s3_ans = s3_resource.Object(bucket, key).put(Body=csv_buffer.getvalue())
    return s3_ans['ResponseMetadata']['HTTPStatusCode'] == HTTPStatus.OK