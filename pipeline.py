from kfp import dsl
from kfp.dsl import Artifact, Input, Output, Model, Dataset, component

# --- 1. CTGAN Data Synthesis Component ---
@component(
    packages_to_install=["sdv", "pandas", "google-cloud-storage"],
    base_image="python:3.9"
)
def synthesize_data(
    model_path: str,
    num_rows: int,
    synthetic_data: Output[Dataset]
):
    import pickle
    import pandas as pd
    from google.cloud import storage

    # Load CTGAN model from GCS
    bucket_name = model_path.split("/")[2]
    blob_name = "/".join(model_path.split("/")[3:])
    
    storage_client = storage.Client()
    blob = storage_client.bucket(bucket_name).blob(blob_name)
    
    with blob.open("rb") as f:
        # Assumes CTGANSynthesizer or similar from SDV
        model = pickle.load(f)
    
    # Generate synthetic samples
    df_synthetic = model.sample(num_rows=num_rows)
    df_synthetic.to_csv(synthetic_data.path, index=False)

# --- 2. VAE Anomaly Detection Component ---
@component(
    packages_to_install=["torch", "pandas", "numpy", "google-cloud-storage"],
    base_image="python:3.9"
)
def vae_feature_extraction(
    model_path: str,
    input_data: Input[Dataset],
    vae_output: Output[Dataset]
):
    import pickle
    import torch
    import pandas as pd
    import numpy as np
    from google.cloud import storage

    # Load VAE model (assumed PyTorch-based stored in pickle)
    bucket_name = model_path.split("/")[2]
    blob_name = "/".join(model_path.split("/")[3:])
    storage_client = storage.Client()
    blob = storage_client.bucket(bucket_name).blob(blob_name)
    
    with blob.open("rb") as f:
        vae_model = pickle.load(f)
    vae_model.eval()
    
    df = pd.read_csv(input_data.path)
    tensor_data = torch.FloatTensor(df.values)
    
    with torch.no_grad():
        recon_batch, mu, logvar = vae_model(tensor_data)
        
        # 1. Reconstruction Error (MSE per sample)
        recon_error = torch.mean((recon_batch - tensor_data)**2, dim=1).numpy()
        
        # 2. KL Divergence Formula: 
        # $$KL(q(z|x) || p(z)) = -0.5 \times \sum(1 + \log(\sigma^2) - \mu^2 - \sigma^2)$$
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).numpy()
    
    # Add new features to the dataframe
    df['recon_error'] = recon_error
    df['kl_divergence'] = kl_div
    df.to_csv(vae_output.path, index=False)

# --- 3. CatBoost Prediction Component ---
@component(
    packages_to_install=["catboost", "pandas", "google-cloud-storage"],
    base_image="python:3.9"
)
def predict_final_value(
    model_path: str,
    engineered_data: Input[Dataset],
    final_results: Output[Dataset]
):
    import pickle
    import pandas as pd
    from google.cloud import storage

    # Load CatBoost model
    bucket_name = model_path.split("/")[2]
    blob_name = "/".join(model_path.split("/")[3:])
    storage_client = storage.Client()
    blob = storage_client.bucket(bucket_name).blob(blob_name)
    
    with blob.open("rb") as f:
        cat_model = pickle.load(f)
    
    df = pd.read_csv(engineered_data.path)
    
    # Predict using the data (including recon_error and kl_divergence)
    df['prediction'] = cat_model.predict(df)
    df.to_csv(final_results.path, index=False)

# --- Pipeline Definition ---
@dsl.pipeline(
    name="3-model-ensemble-pipeline",
    pipeline_root="gs://your-project-bucket/pipeline_root"
)
def multi_model_pipeline(
    ctgan_uri: str,
    vae_uri: str,
    catboost_uri: str,
    num_samples: int = 5000
):
    step1 = synthesize_data(model_path=ctgan_uri, num_rows=num_samples)
    
    step2 = vae_feature_extraction(
        model_path=vae_uri, 
        input_data=step1.outputs['synthetic_data']
    )
    
    step3 = predict_final_value(
        model_path=catboost_uri, 
        engineered_data=step2.outputs['vae_output']
    )