import functions_framework
from PyPDF2 import PdfReader
from google.cloud import secretmanager
import google_crc32c
import json
from google.cloud import storage
import pandas as pd


@functions_framework.http
def extract_text(request):

    request_json = request.get_json(silent=True)
    request_args = request.args
    secret_client = secretmanager.SecretManagerServiceClient()
    name = f"projects/84798369686/secrets/authkeys/versions/1"
    
    response = secret_client.access_secret_version(request={"name": name})
    payload = response.payload.data.decode("UTF-8")
    credentials = json.loads(payload)
    storage_client = storage.Client()
    bucket = storage_client.bucket('pdfs-for-text-extraction')
    blobs = storage_client.list_blobs('pdfs-for-text-extraction')
    dir_list=[]
    i=0
    for blob in blobs:
        if 'data_for_fm' not in blob.name:
            blob.download_to_filename(f"/tmp/data_{str(i)}.pdf")
            dir_list.append(f"/tmp/data_{str(i)}.pdf")
        i+=1
    overrall_text=''
    for j in dir_list:
        reader = PdfReader(j)
        num_pages = len(reader.pages)
        for k in range(num_pages):
            page = reader.pages[k]
            text = page.extract_text()
            overrall_text = overrall_text+" "+text
    request_json = request.get_json(silent=True)
    request_args = request.args
    with open("/tmp/data_for_fm.txt", "w") as f:
        f.write(overrall_text)
    blob = bucket.blob("data_for_fm.txt")
    blob.upload_from_filename("/tmp/data_for_fm.txt")

    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'name' in request_json:
        name = request_json['name']
    elif request_args and 'name' in request_args:
        name = request_args['name']
    else:
        name = 'World'
    return 'Hello {}!'.format(name)