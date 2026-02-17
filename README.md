MultiCloud Anomaly Detection system in Cloud Billing Cost <br/>
Sanchit Mishra <br/>
MSc - Artificial Intelligence and Machine Learning <br/>

Problem Definition: As more and more enterprises are migrating their data from on-prem system to cloud environment, the use of cloud computing is scaling up rapidly. But this use is sometimes followed by an unprecedented high cloud cost usage due to uncontrolled use of resources, so there is a need to govern and predict such unprecedented high cost and taking steps to prevent such anomalies. This can be done by implementing anomaly detection system can oversee the predicted high cost in future by using Unsupervised Learning Algorithms and implement steps to prevent or control the cloud resources using LLM agents that can frame cloud commands.

For the sake of ongoing study the information about model architecture cannot be disclosed but there are basically two models one is Variational Autoencoder based model used to assign pseudo labels based on threshold and other model is Classification Model Feeding on the labelled output from downstream model

1. AWS Ingestion Path
Source: AWS Cost and Usage Report (CUR).

Trigger: An AWS Lambda function processes the reports.

Storage: Data is temporarily staged in an Amazon S3 Standard bucket before being pulled into the GCP environment.

2. Azure Ingestion Path
Source: Azure Cost Management and Billing.

Storage: Data is exported to an Azure Storage Container (Blob Storage).

3. GCP Ingestion Path
Source: Cloud Billing API.

Direct Ingestion: Data is streamed or loaded directly into the BigQuery environment.

4. Processing & Centralization
Google Cloud Functions: Acts as the primary orchestrator and transformer. It fetches the staged data from AWS S3 and Azure Storage, standardizes the schema, and loads it into the destination.

Destination: Google BigQuery, enabling cross-cloud cost analysis and visualization (e.g., via Looker or Data Studio).


