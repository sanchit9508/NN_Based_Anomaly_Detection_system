import json
import boto3
import pandas as pd
import os


def lambda_handler(event, context):
    
    s3 = boto3.client('s3')
    bucket_name = 'finops55'
    listofobject=s3.list_objects_v2(Bucket=bucket_name)['Contents']
    df=pd.DataFrame()
    for i in listofobject:
        print(i['Key'])
        if 'aws_data/usage_and_cost_data/data' in i['Key'] and '.csv.gz' in i['Key']:
            s3_key = str(i['Key'])
            print(s3_key)
            local_file_path = '/tmp/' + s3_key
            # print(local_file_path)
            s3.download_file(bucket_name, s3_key,'/tmp/data.csv')
            # print(pd.read_csv('/tmp/data.csv',compression='gzip').shape)
            df=pd.concat([df,pd.read_csv('/tmp/data.csv',compression='gzip')])
            os.remove('/tmp/data.csv')
    df.to_csv('/tmp/aws_usage_and_cost_data-00001.csv')
    print(df.shape)
    s3.upload_file('/tmp/aws_usage_and_cost_data-00001.csv', bucket_name, 'aws_usage_and_cost_data-00001.csv')
    df=pd.DataFrame()
    for i in listofobject:
        if 'focus_cost/aws-focus/' in i['Key'] and '.csv.gz' in i['Key']:
                s3_key = str(i['Key'])
                print(s3_key)
                local_file_path = '/tmp/' + s3_key
                # print(local_file_path)
                s3.download_file(bucket_name, s3_key,'/tmp/data.csv')
                # print(pd.read_csv('/tmp/data.csv',compression='gzip').shape)
                df=pd.concat([df,pd.read_csv('/tmp/data.csv',compression='gzip')])
                os.remove('/tmp/data.csv')
    df.to_csv('/tmp/aws_focus_usage_and_cost_data-00001.csv')
    print(df.shape)
    s3.upload_file('/tmp/aws_focus_usage_and_cost_data-00001.csv', bucket_name, 'aws_focus_usage_and_cost_data-00001.csv')

    return 'finished '
