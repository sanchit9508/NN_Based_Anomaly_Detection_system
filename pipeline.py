
from kfp import dsl, compiler
from kfp.client import Client
from kfp.dsl import Dataset, Model, Metrics, Output, Input


@dsl.component(
    base_image='python:3.10.11',
    packages_to_install=['pandas==2.0.3', 'scikit-learn==1.3.0']
)
def load_data(
    n_samples: int,
    n_features: int,
    output_data: Output[Dataset]
):

    import pandas as pd
    
    print(f" Generating dataset: {n_samples} samples, {n_features} features")
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        n_redundant=int(n_features * 0.2),
        n_classes=2,
        random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Save to artifact path (KFP provides this path automatically)
    # In Airflow, you'd manually manage this path
    df.to_csv(output_data.path, index=False, encoding='utf-8')
    
    # Add metadata - this appears in KFP UI and enables search/filtering
    # In Airflow, you'd need MLflow or custom logging for this
    output_data.metadata['num_samples'] = n_samples
    output_data.metadata['num_features'] = n_features
    output_data.metadata['target_distribution'] = str(df['target'].value_counts().to_dict())
    
    print(f" Data saved: {len(df)} rows")
    print(f"   Target distribution: {df['target'].value_counts().to_dict()}")

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.0.3', 'scikit-learn==1.3.0']
)
def preprocess_data(
    input_data: Input[Dataset],
    test_size: float,
    train_data: Output[Dataset],
    test_data: Output[Dataset]
):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Read from artifact path (KFP downloaded this from MinIO automatically)
    print(f" Loading data from previous step...")
    df = pd.read_csv(input_data.path)
    print(f"   Loaded {len(df)} rows")
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    print(f" Splitting: {(1-test_size)*100:.0f}% train, {test_size*100:.0f}% test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Scale features
    print(" Applying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create output DataFrames
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df['target'] = y_train.values
    
    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df['target'] = y_test.values
    
    # Save to artifact paths
    train_df.to_csv(train_data.path, index=False, encoding='utf-8')
    test_df.to_csv(test_data.path, index=False, encoding='utf-8')
    
    # Add metadata
    train_data.metadata['samples'] = len(train_df)
    train_data.metadata['scaled'] = True
    test_data.metadata['samples'] = len(test_df)
    test_data.metadata['scaled'] = True
    
    print(f" Train set: {len(train_df)} samples")
    print(f" Test set: {len(test_df)} samples")


@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.0.3', 'scikit-learn==1.3.0', 'joblib==1.3.2']
)
def train_model(
    train_data: Input[Dataset],
    n_estimators: int,
    max_depth: int,
    model_output: Output[Model]
):
    """Train a RandomForest classifier.
    
    Args:
        train_data: Training Dataset from preprocessing
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of each tree
        model_output: Output Model artifact
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    
    # Load training data
    print(f" Loading training data...")
    df = pd.read_csv(train_data.path)
    X_train = df.drop('target', axis=1)
    y_train = df['target']
    print(f"   Training samples: {len(X_train)}")
    
    # Train model
    print(f" Training RandomForest:")
    print(f"   n_estimators: {n_estimators}")
    print(f"   max_depth: {max_depth}")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    model.fit(X_train, y_train)
    
    # Calculate training accuracy
    train_accuracy = model.score(X_train, y_train)
    print(f" Training accuracy: {train_accuracy:.4f}")
    
    # Save model to artifact path
    joblib.dump(model, model_output.path)
    
    # Add rich metadata - this is automatically tracked!
    # In Airflow, you'd need MLflow for this level of tracking
    model_output.metadata['framework'] = 'sklearn'
    model_output.metadata['algorithm'] = 'RandomForestClassifier'
    model_output.metadata['n_estimators'] = n_estimators
    model_output.metadata['max_depth'] = max_depth
    model_output.metadata['train_accuracy'] = float(train_accuracy)
    model_output.metadata['n_features'] = X_train.shape[1]
    
    print(f" Model saved with metadata")

@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==2.0.3', 'scikit-learn==1.3.0', 'joblib==1.3.2']
)
def evaluate_model(
    model_input: Input[Model],
    test_data: Input[Dataset],
    metrics: Output[Metrics]
) -> float:
    
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import joblib
    
    # Load model
    print(f" Loading model...")
    model = joblib.load(model_input.path)
    
    # Load test data
    print(f" Loading test data...")
    df = pd.read_csv(test_data.path)
    X_test = df.drop('target', axis=1)
    y_test = df['target']
    print(f"   Test samples: {len(X_test)}")
    
    # Make predictions
    print(f" Making predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Log metrics to artifact - THESE APPEAR IN KFP UI!
    # In Airflow, you'd need MLflow or custom dashboards
    metrics.log_metric('accuracy', float(accuracy))
    metrics.log_metric('precision', float(precision))
    metrics.log_metric('recall', float(recall))
    metrics.log_metric('f1_score', float(f1))
    
    # Print results (appears in pod logs)
    print("\n" + "=" * 50)
    print(" MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    print("=" * 50)
    
    # Return accuracy - can be used for conditional deployment
    return float(accuracy)

@dsl.pipeline(
    name='ml-training-pipeline',
    description='Complete ML pipeline: Load → Preprocess → Train → Evaluate'
)
def ml_training_pipeline(
    n_samples: int = 1000,
    n_features: int = 20,
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: int = 10
):
    
    load_task = load_data(
        n_samples=n_samples,
        n_features=n_features
    )
    
    preprocess_task = preprocess_data(
        input_data=load_task.outputs['output_data'],  # Automatic dependency!
        test_size=test_size
    )
    
    train_task = train_model(
        train_data=preprocess_task.outputs['train_data'],
        n_estimators=n_estimators,
        max_depth=max_depth
    )
    
    evaluate_task = evaluate_model(
        model_input=train_task.outputs['model_output'],
        test_data=preprocess_task.outputs['test_data']
    )

if __name__ == '__main__':
    import sys
    
    # Compile the pipeline
    print("=" * 60)
    print("KUBEFLOW PIPELINES - ML TRAINING PIPELINE")
    print("=" * 60)
    
    print("\n Step 1: Compiling pipeline to YAML...")
    compiler.Compiler().compile(
        pipeline_func=ml_training_pipeline,
        package_path='ml_training_pipeline.yaml'
    )
    print("    Compiled to: ml_training_pipeline.yaml")
    
    # Connect to KFP
    print("\n Step 2: Connecting to Kubeflow Pipelines...")
    try:
        client = Client(host='http://localhost:8888')
        print("    Connected to KFP API")
    except Exception as e:
        print(f"    Connection failed: {e}")
        print("   Make sure port-forward is running:")
        print("   kubectl port-forward -n kubeflow svc/ml-pipeline 8888:8888 --address 0.0.0.0 &")
        sys.exit(1)
    
    # Submit the pipeline
    print("\n Step 3: Submitting pipeline...")
    run = client.create_run_from_pipeline_package(
        pipeline_file='ml_training_pipeline.yaml',
        arguments={
            'n_samples': 1000,
            'n_features': 20,
            'test_size': 0.2,
            'n_estimators': 100,
            'max_depth': 10
        },
        run_name='ml-training-run',
        experiment_name='ml-training-experiments'
    )
    
    print(f"    Run submitted!")
    run_id = getattr(run, 'run_id', None) or getattr(run, 'id', None) or run
    print(f"   Run ID: {run_id}")
    print("\n" + "=" * 60)
    print("WHAT HAPPENS NEXT:")
    print("=" * 60)
