import functions_framework
from google.cloud import storage
import boto3
import pandas as pd
import pandas_gbq
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient
from google.cloud import secretmanager
import google_crc32c
import json

def filter_and_get_azure(credentials,type_az):
    blob_service_client = BlobServiceClient.from_connection_string(credentials['azure_connection_string'])
    container_client = blob_service_client.get_container_client("billexport")
    blob_list = container_client.list_blobs()
    blob_list = container_client.list_blobs()
    blob_list_ini = [j['name'] for j in blob_list]
    print(blob_list_ini)
    if type_az == 'cost':
        filtered_blob = list(filter(lambda x:('part_0_0001.csv.gz' in x) and ('cost/usage_cost-actual-cost/' in x),blob_list_ini))
    elif type_az=='focus':
        filtered_blob = list(filter(lambda x:('part_0_0001.csv' in x) and ('focusdata/azure-focus-cost/' in x),blob_list_ini))
    print(filtered_blob)
    df=pd.DataFrame()
    for i in filtered_blob:
        print(i)
        blob_client_1 = container_client.get_blob_client(i)
        print(blob_client_1)
        download_file_path_actual = f"/tmp/"+str(i.split('/')[-1])
        with open(download_file_path_actual, "wb") as download_file:
            download_file.write(blob_client_1.download_blob().readall()) 
        if type_az == 'cost':
            df1=pd.read_csv(download_file_path_actual,compression="gzip")
        else:
            df1=pd.read_csv(download_file_path_actual)
        df=pd.concat([df,df1])
    return df




@functions_framework.http
def hello_http(request):
    request_json = request.get_json(silent=True)
    request_args = request.args
    secret_client = secretmanager.SecretManagerServiceClient()
    name = f"projects/84798369686/secrets/authkeys/versions/1"
    
    response = secret_client.access_secret_version(request={"name": name})
    payload = response.payload.data.decode("UTF-8")
    credentials = json.loads(payload)
    storage_client = storage.Client()
    bucket = storage_client.bucket('finops_export_csvs')

    #AWS
    s3 = boto3.client('s3', aws_access_key_id=credentials["aws_access_id"], aws_secret_access_key=credentials["aws_secret"])
    bucket_name = 'finops55'
    s3_key = 'aws_usage_and_cost_data-00001.csv'
    s3_focus_key='aws_focus_usage_and_cost_data-00001.csv'
    #s3_key = 'aws_data/usage_and_cost_data/data/BILLING_PERIOD=2025-07/usage_and_cost_data-00001.csv.gz'  
    local_file_path = '/tmp/aws_usage_and_cost_data-00001.csv' 
    local_focus_file_path = '/tmp/aws_focus_usage_and_cost_data-00001.csv' 
    s3.download_file(bucket_name, s3_key, local_file_path)
    s3.download_file(bucket_name, s3_focus_key, local_focus_file_path)
    print(f"Downloaded aws to {local_file_path}")
    #AWS_Actual
    blob_aws = bucket.blob('aws_usage_and_cost_data-00001.csv')
    df = pd.read_csv('/tmp/aws_usage_and_cost_data-00001.csv')
    df.to_csv('/tmp/aws_usage_and_cost_data-00001.csv',index=False)
    blob_aws.upload_from_filename('/tmp/aws_usage_and_cost_data-00001.csv')
    pandas_gbq.to_gbq(df,'finops-55.billing_1.aws_usage_and_cost',project_id='finops-55',
        if_exists="replace")
    #AWS_Focus
    blob_focus_aws = bucket.blob('aws_focus_usage_and_cost_data-00001.csv')
    df = pd.read_csv('/tmp/aws_focus_usage_and_cost_data-00001.csv')
    df.to_csv('/tmp/aws_focus_usage_and_cost_data-00001.csv',index=False)
    blob_focus_aws.upload_from_filename('/tmp/aws_focus_usage_and_cost_data-00001.csv')
    pandas_gbq.to_gbq(df,'finops-55.billing_1.aws_focus_usage_and_cost',project_id='finops-55',
        if_exists="replace")
    print(f"exported aws data to gcs and bigquery")



    #Azure Actual
    df_az_actual = filter_and_get_azure(credentials,type_az='cost')
    df_az_actual.to_csv('/tmp/az_actual_part_0_0001.csv',index=False)
    blob_az_actual = bucket.blob('az_actual_part_0_0001.csv')
    blob_az_actual.upload_from_filename('/tmp/az_actual_part_0_0001.csv')
    pandas_gbq.to_gbq(df_az_actual,'finops-55.billing_1.azure_actual_usage_and_cost',project_id='finops-55',if_exists="replace")
    print(f"exported azure actual data to gcs and bigquery")

    #Azure Focus
    df_az_focus = filter_and_get_azure(credentials,type_az='focus')
    df_az_focus.to_csv('/tmp/az_focus_part_0_0001.csv',index=False)  
    blob = bucket.blob('az_focus_part_0_0001.csv')
    blob.upload_from_filename('/tmp/az_focus_part_0_0001.csv')
    pandas_gbq.to_gbq(df_az_focus,'finops-55.billing_1.azure_focus_usage_and_cost',project_id='finops-55',if_exists="replace")
    print(f"exported azure focus data to gcs and bigquery")
    return 'Finished'