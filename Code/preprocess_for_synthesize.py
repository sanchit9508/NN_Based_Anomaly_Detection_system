import numpy as np
import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account
from datetime import datetime
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

def dataclean_for_syn(df_gcp_focus,df_aws_focus,df_azure_focus):
    def convert_datetime(df):
        df['chargeperiodstart'] = pd.to_datetime(df['chargeperiodstart'])
        df['chargeperiodend'] = pd.to_datetime(df['chargeperiodend'])
        df['billingperiodend'] = pd.to_datetime(df['billingperiodend'])
        df['billingperiodstart'] = pd.to_datetime(df['billingperiodstart'])
        df['chargeperiodstart_year'] = df['chargeperiodstart'].dt.year
        df['chargeperiodend_year'] = df['chargeperiodend'].dt.year
        df['billingperiodstart_year'] = df['billingperiodstart'].dt.year
        df['billingperiodend_year'] = df['billingperiodend'].dt.year
        df['billingperiodend_month'] = df['billingperiodend'].dt.month
        df['billingperiodstart_month'] = df['billingperiodstart'].dt.month
        df['chargeperiodstart_month'] = df['chargeperiodstart'].dt.month
        df['chargeperiodend_month'] = df['chargeperiodend'].dt.month
        df['chargeperiodend_day'] = df['chargeperiodend'].dt.day
        df['chargeperiodstart_day'] = df['chargeperiodstart'].dt.day
        #df['billingperiodstart_day'] = df['billingperiodstart'].dt.day
        #df['billingperiodend_day'] = df['billingperiodend'].dt.day
        #df['chargeperiodend_hour'] = df['chargeperiodend'].dt.hour
        #df['chargeperiodstart_hour'] = df['chargeperiodstart'].dt.hour
        #df['billingperiodstart_hour'] = df['billingperiodstart'].dt.hour
        #df['billingperiodend_hour'] = df['billingperiodend'].dt.hour
        #df['chargeperiodend_min'] = df['chargeperiodend'].dt.minute
        #df['chargeperiodstart_min'] = df['chargeperiodstart'].dt.minute
        #df['billingperiodstart_min'] = df['billingperiodstart'].dt.minute
        #df['billingperiodend_min'] = df['billingperiodend'].dt.minute
        df=df.drop(['chargeperiodstart','chargeperiodend','billingperiodend','billingperiodstart'],axis=1)
        for i in ['billedcost', 'contractedcost', 'contractedunitprice', 'effectivecost', 'listcost',
        'listunitprice', 'pricingquantity','consumedquantity']:
            df[i]=df[i].astype(np.number)
        return df
    df_aws_focus = convert_datetime(df_aws_focus)
    df_gcp_focus = convert_datetime(df_gcp_focus)
    df_azure_focus = convert_datetime(df_azure_focus)
    b=df_aws_focus.columns
    b=[i.lower() for i in b]
    c=df_azure_focus.columns
    c=[i.lower() for i in c]
    col_gcp=pd.DataFrame({'GCP_Columns':a})
    col_aws=pd.DataFrame({'AWS_Columns':b})
    col_azure=pd.DataFrame({'AZURE_Columns':c})
    col_total=col_gcp.merge(col_aws,how="inner",left_on="GCP_Columns",right_on="AWS_Columns").merge(col_azure,how="inner",left_on="GCP_Columns",right_on="AZURE_Columns")
    df_gcp_focus_1 = df_gcp_focus[col_total['GCP_Columns'].tolist()]
    #Removing id columns
    id_columns=[]
    for i in df_gcp_focus_1.columns:
        if 'id' in i[-2:]:
            id_columns.append(i)
    id_columns
    df_gcp_focus_1.drop(id_columns,axis=1,inplace=True)
    #Removing Redundant columns
    df_gcp_focus_1.drop(['providername','providername'],axis=1,inplace=True)
    df_aws_focus_1 = df_aws_focus[col_total['AWS_Columns'].tolist()]
    #Removing id columns
    id_columns=[]
    for i in df_aws_focus_1.columns:
        if 'id' in i[-2:]:
            id_columns.append(i)
    id_columns
    df_aws_focus_1.drop(id_columns,axis=1,inplace=True)
    df_aws_focus_1.isnull().sum()/len(df_aws_focus_1)*100
    #Removing Redundant columns
    df_aws_focus_1.drop(['providername','providername'],axis=1,inplace=True)
    df_azure_focus_1 = df_azure_focus[col_total['AZURE_Columns'].tolist()]
    #Removing id columns
    id_columns=[]
    for i in df_azure_focus_1.columns:
        if 'id' in i[-2:]:
            id_columns.append(i)
    id_columns
    df_azure_focus_1.drop(id_columns,axis=1,inplace=True)
    #Removing Redundant columns
    df_azure_focus_1.drop(['providername','providername'],axis=1,inplace=True)

    df_azure_focus_1.isnull().sum()/len(df_azure_focus_1)*100
    df_gcp_focus_1.drop('tags',axis=1,inplace=True)
    df_aws_focus_1.drop('tags',axis=1,inplace=True)
    df_azure_focus_1.drop('tags',axis=1,inplace=True)
    df_full = pd.concat([df_gcp_focus_1,df_aws_focus_1,df_azure_focus_1])
    df_full.isnull().sum()/len(df_full)*100
    null_cols=[]
    for i in df_full.columns:
        if df_full[i].isnull().sum()/len(df_full)*100>40:
            null_cols.append(i)
    df_aws_focus_1.drop(null_cols,axis=1,inplace=True)
    df_gcp_focus_1.drop(null_cols,axis=1,inplace=True)
    df_azure_focus_1.drop(null_cols,axis=1,inplace=True)
    list1=df_gcp_focus_1.columns.tolist()
    list2=['billedcost','effectivecost',  'consumedquantity', 'pricingquantity','listcost','contractedcost']
    for i in list2:
        list1.remove(i)
    list1
    df_gcp_focus_2 = df_gcp_focus_1.groupby(list1)[list2].sum().reset_index()
    df_aws_focus_2 = df_aws_focus_1.groupby(list1)[list2].sum().reset_index()
    df_azure_focus_2 = df_azure_focus_1.groupby(list1)[list2].sum().reset_index()
    return df_gcp_focus_2,df_azure_focus_2,df_aws_focus_2




