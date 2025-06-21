import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import polars as pl
import pathlib
import time
from collections import defaultdict
import pickle
import os
from pathlib import Path
import logging
import json
from copy import deepcopy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- PyTorch Dataset for Tabular Data (Simplified) ---
class TabularDataset(Dataset):
    def __init__(self, X_categorical, X_continuous, y):
        self.X_categorical = X_categorical
        self.X_continuous = X_continuous
        self.y = y

    def __len__(self):
        # Determine length primarily from y, assuming it's always present for supervised learning
        if torch.is_tensor(self.y) and self.y.ndim > 0:
            return self.y.size(0)
        # Fallback if y is somehow not definitive (e.g. unsupervised or problematic input)
        elif torch.is_tensor(self.X_categorical) and self.X_categorical.ndim > 0 and self.X_categorical.size(0) > 0:
            return self.X_categorical.size(0)
        elif torch.is_tensor(self.X_continuous) and self.X_continuous.ndim > 0 and self.X_continuous.size(0) > 0:
            return self.X_continuous.size(0)
        # If all are empty or not tensors with size, dataset is empty
        # logger.warning("TabularDataset has no determinable length from y, X_categorical, or X_continuous.")
        return 0

    def __getitem__(self, idx):
        # Ensure tensors are returned even if source tensors are empty (correct shape for DataLoader)
        cat_item = self.X_categorical[idx] if torch.is_tensor(self.X_categorical) and self.X_categorical.numel() > 0 and self.X_categorical.size(0) > idx else torch.empty(0, dtype=torch.long)
        cont_item = self.X_continuous[idx] if torch.is_tensor(self.X_continuous) and self.X_continuous.numel() > 0 and self.X_continuous.size(0) > idx else torch.empty(0, dtype=torch.float32)
        # y should always exist for supervised learning, but handle defensively
        label_item = self.y[idx] if torch.is_tensor(self.y) and self.y.numel() > 0 and self.y.size(0) > idx else torch.empty(0, dtype=torch.long)
        
        return {
            'categorical': cat_item,
            'continuous': cont_item,
            'labels': label_item
        }

# --- PyTorch Model Definition ---
class TabularNNModel(nn.Module):
    def __init__(self, embedding_sizes, n_continuous, n_classes, layers, p_dropout=0.1):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        n_embeddings = sum(e.embedding_dim for e in self.embeddings)
        self.n_continuous = n_continuous
        
        all_layers = []
        input_size = n_embeddings + n_continuous
        
        for i, layer_size in enumerate(layers):
            all_layers.append(nn.Linear(input_size, layer_size))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(layer_size))
            all_layers.append(nn.Dropout(p_dropout))
            input_size = layer_size
            
        all_layers.append(nn.Linear(layers[-1], n_classes))
        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_categorical, x_continuous):
        x_embeddings = []
        for i, e in enumerate(self.embeddings):
            x_embeddings.append(e(x_categorical[:, i]))
        x = torch.cat(x_embeddings, 1)
        
        if self.n_continuous > 0:
            if x_continuous.ndim == 1: # Add batch dimension if missing
                 x_continuous = x_continuous.unsqueeze(0)
            x = torch.cat([x, x_continuous], 1)
            
        x = self.layers(x)
        return x

# --- Training Function ---
def train_pytorch_model(
    df_train_full: pl.DataFrame,
    target_name: str,
    game_state_columns: dict,
    model_save_path_full: str,
    nsamples: int = -1,
    cat_embed_dim_multiplier: int = 50,
    min_cat_embed_dim: int = 4,
    max_cat_embed_dim: int = 300,
    layers_config: list = [200, 100],
    p_dropout: float = 0.1,
    lr: float = 1e-3,
    epochs: int = 20,
    batch_size: int = 64,
    early_stopping_patience: int = 3,
    random_state: int = 42,
    stratify_split: bool = True,
    test_size: float = 0.2
):
    logger.info(f"Starting PyTorch model training. Target: {target_name}, Save path: {model_save_path_full}")
    
    if nsamples > 0 and nsamples < len(df_train_full):
        logger.info(f"Sampling {nsamples} from the training data.")
        df_train_full = df_train_full.sample(n=nsamples, seed=random_state)

    df_pandas = df_train_full.to_pandas()

    # Identify categorical and continuous features
    categorical_feature_names = []
    continuous_feature_names = []

    if game_state_columns and 'categorical' in game_state_columns and 'continuous' in game_state_columns:
        categorical_feature_names = [col for col in game_state_columns['categorical'] if col in df_pandas.columns and col != target_name]
        continuous_feature_names = [col for col in game_state_columns['continuous'] if col in df_pandas.columns and col != target_name]
        
        # Ensure all columns (except target) are covered, if not, infer from dtype
        all_known_features = set(categorical_feature_names + continuous_feature_names)
        for col in df_pandas.columns:
            if col != target_name and col not in all_known_features:
                if pd.api.types.is_numeric_dtype(df_pandas[col]):
                    logger.warning(f"Column '{col}' not in game_state_columns, inferring as continuous.")
                    continuous_feature_names.append(col)
                else:
                    logger.warning(f"Column '{col}' not in game_state_columns, inferring as categorical.")
                    categorical_feature_names.append(col)
    else: # Fallback to dtype inference if game_state_columns is not detailed enough
        logger.warning("game_state_columns not fully provided or missing 'categorical'/'continuous' keys. Inferring feature types from pandas dtypes.")
        for col in df_pandas.columns:
            if col == target_name:
                continue
            if pd.api.types.is_numeric_dtype(df_pandas[col]): # and df_pandas[col].nunique() > 10: # Heuristic for continuous
                continuous_feature_names.append(col)
            else:
                categorical_feature_names.append(col)
    
    logger.info(f"Categorical features: {categorical_feature_names}")
    logger.info(f"Continuous features: {continuous_feature_names}")

    # Preprocessing
    artifacts = {
        'target_encoder': LabelEncoder(),
        'categorical_encoders': {col: LabelEncoder() for col in categorical_feature_names},
        'continuous_scalers': {col: StandardScaler() for col in continuous_feature_names},
        'na_fills': {},
        'categorical_feature_names': categorical_feature_names,
        'continuous_feature_names': continuous_feature_names,
        'target_name': target_name,
        'model_params': {},
        'training_history': {}
    }

    # Target encoding
    df_pandas[target_name] = df_pandas[target_name].astype(str) # Ensure target is string for robust encoding
    artifacts['target_encoder'].fit(df_pandas[target_name])
    df_pandas[target_name] = artifacts['target_encoder'].transform(df_pandas[target_name])
    n_classes = len(artifacts['target_encoder'].classes_)
    logger.info(f"Target '{target_name}' encoded. Number of classes: {n_classes}")

    # Categorical feature encoding and NA handling
    for col in categorical_feature_names:
        df_pandas[col] = df_pandas[col].astype(str) # Ensure consistent type
        fill_val_cat = "MISSING_CAT" 
        artifacts['na_fills'][col] = fill_val_cat

        # Ensure MISSING_CAT is part of the classes known to the encoder
        unique_values_for_fit = pd.unique(df_pandas[col].fillna(fill_val_cat)).tolist()
        if fill_val_cat not in unique_values_for_fit:
            unique_values_for_fit.append(fill_val_cat)
        
        artifacts['categorical_encoders'][col].fit(unique_values_for_fit)
        
        # Now transform the column. Any original NaNs are already fill_val_cat.
        # Any values not in unique_values_for_fit during .fit() would cause error here if not handled by prior fillna
        # This assumes all values in df_pandas[col] after fillna are now covered by the fit.
        df_pandas[col] = artifacts['categorical_encoders'][col].transform(df_pandas[col].fillna(fill_val_cat))

    # Continuous feature scaling and NA handling
    for col in continuous_feature_names:
        fill_val_cont = df_pandas[col].median() # Use median for NA fill for continuous
        artifacts['na_fills'][col] = fill_val_cont
        df_pandas[col] = df_pandas[col].fillna(fill_val_cont)
        df_pandas[col] = artifacts['continuous_scalers'][col].fit_transform(df_pandas[col].values.reshape(-1, 1))

    # Calculate embedding sizes
    embedding_sizes = []
    for col in categorical_feature_names:
        num_categories = len(artifacts['categorical_encoders'][col].classes_)
        embed_dim = max(min_cat_embed_dim, min(max_cat_embed_dim, int(num_categories / 2) if cat_embed_dim_multiplier == -1 else cat_embed_dim_multiplier)) # Original fastai logic
        embedding_sizes.append((num_categories, embed_dim))
    
    artifacts['embedding_sizes'] = embedding_sizes
    logger.info(f"Embedding sizes: {embedding_sizes}")

    # Prepare data for PyTorch
    X_cat = torch.tensor(df_pandas[categorical_feature_names].values, dtype=torch.long) if categorical_feature_names else torch.empty(len(df_pandas), 0, dtype=torch.long)
    X_cont = torch.tensor(df_pandas[continuous_feature_names].values, dtype=torch.float32) if continuous_feature_names else torch.empty(len(df_pandas), 0, dtype=torch.float32)
    y = torch.tensor(df_pandas[target_name].values, dtype=torch.long)

    # Train/validation split
    stratify_col = y if stratify_split and n_classes > 1 else None
    try:
        if stratify_col is not None and len(torch.unique(stratify_col)) < 2: # Not enough classes to stratify
             logger.warning("Stratification requested but not enough classes in target for stratification. Proceeding without it.")
             stratify_col = None
        
        indices = np.arange(len(y))
        train_indices, val_indices = train_test_split(indices, test_size=test_size, random_state=random_state, stratify=stratify_col.numpy() if stratify_col is not None else None)
    except ValueError as e:
        logger.warning(f"Stratified split failed: {e}. Falling back to non-stratified split.")
        train_indices, val_indices = train_test_split(indices, test_size=test_size, random_state=random_state)

    X_cat_train, X_cat_val = X_cat[train_indices], X_cat[val_indices]
    X_cont_train, X_cont_val = X_cont[train_indices], X_cont[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    train_dataset = TabularDataset(X_cat_train, X_cont_train, y_train)
    val_dataset = TabularDataset(X_cat_val, X_cont_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model, Loss, Optimizer, Scheduler
    n_continuous_features = X_cont.shape[1]
    model = TabularNNModel(embedding_sizes, n_continuous_features, n_classes, layers_config, p_dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)

    artifacts['model_params'] = {
        'embedding_sizes': embedding_sizes,
        'n_continuous': n_continuous_features,
        'n_classes': n_classes,
        'layers': layers_config,
        'p_dropout': p_dropout
    }
    
    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}

    logger.info(f"Starting PyTorch training loop for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions_train = 0
        total_predictions_train = 0

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(batch['categorical'], batch['continuous'])
            loss = criterion(outputs, batch['labels'])
            loss.backward()
            optimizer.step()
            scheduler.step() # Step OneCycleLR scheduler each batch

            running_loss += loss.item() * batch['labels'].size(0) # Weighted by batch size
            _, predicted = torch.max(outputs.data, 1)
            total_predictions_train += batch['labels'].size(0)
            correct_predictions_train += (predicted == batch['labels']).sum().item()
            
            if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)} - Train Loss: {loss.item():.4f}, LR: {current_lr:.3e}")
        
        epoch_loss_train = running_loss / len(train_loader.dataset)
        epoch_acc_train = correct_predictions_train / total_predictions_train
        history['train_loss'].append(epoch_loss_train)
        history['train_acc'].append(epoch_acc_train)
        history['lr'].append(optimizer.param_groups[0]['lr']) # Log LR at end of epoch

        # Validation
        model.eval()
        val_loss = 0.0
        correct_predictions_val = 0
        total_predictions_val = 0
        with torch.no_grad():
            for i_val, batch_val in enumerate(val_loader):
                outputs_val = model(batch_val['categorical'], batch_val['continuous'])
                loss_val = criterion(outputs_val, batch_val['labels'])
                val_loss += loss_val.item() * batch_val['labels'].size(0) # Weighted by batch size
                _, predicted_val = torch.max(outputs_val.data, 1)
                total_predictions_val += batch_val['labels'].size(0)
                correct_predictions_val += (predicted_val == batch_val['labels']).sum().item()
                if (i_val + 1) % 50 == 0 or (i_val + 1) == len(val_loader):
                    logger.info(f"Epoch {epoch+1}/{epochs} - Validating Batch {i_val+1}/{len(val_loader)}")

        epoch_loss_val = val_loss / len(val_loader.dataset)
        epoch_acc_val = correct_predictions_val / total_predictions_val
        history['val_loss'].append(epoch_loss_val)
        history['val_acc'].append(epoch_acc_val)
        
        current_lr_display = history['lr'][-1] # Get the last recorded LR for this epoch
        logger.info(f"Epoch {epoch+1}/{epochs} - LR: {current_lr_display:.3e} - Train Loss: {epoch_loss_train:.4f}, Acc: {epoch_acc_train:.4f} | Val Loss: {epoch_loss_val:.4f}, Acc: {epoch_acc_val:.4f}")

        if epoch_loss_val < best_val_loss:
            best_val_loss = epoch_loss_val
            epochs_no_improve = 0
            
            base_save_path = Path(model_save_path_full) # Expects "dir/basename"
            model_file_with_pth = base_save_path.with_suffix('.pth')
            artifacts_file_with_pkl = base_save_path.with_suffix('.pkl')
            
            model_file_with_pth.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            
            torch.save(model.state_dict(), str(model_file_with_pth))
            # Save artifacts associated with this best model
            current_artifacts_for_best_model = artifacts.copy() # Avoid modifying the main artifacts dict directly here
            current_artifacts_for_best_model['best_model_epoch'] = epoch + 1
            current_artifacts_for_best_model['best_model_val_loss'] = best_val_loss
            with open(artifacts_file_with_pkl, 'wb') as f:
                pickle.dump(current_artifacts_for_best_model, f) # Save a snapshot of artifacts for this best model
            
            logger.info(f"Validation loss improved to {best_val_loss:.6f}. Storing model state at {model_file_with_pth} and associated artifacts at {artifacts_file_with_pkl}")
            # Update main artifacts dict with best epoch info
            artifacts['best_model_epoch'] = epoch + 1
            artifacts['best_model_val_loss'] = best_val_loss
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs.")
            # Path to the best saved model
            best_model_path_to_load = Path(model_save_path_full).with_suffix('.pth')
            if best_model_path_to_load.exists():
                model.load_state_dict(torch.load(str(best_model_path_to_load)))
                logger.info(f"Loaded best model state from {best_model_path_to_load} (epoch {artifacts.get('best_model_epoch', 'N/A')} with val_loss: {artifacts.get('best_model_val_loss', 'N/A')})")
            else:
                logger.warning(f"Early stopping triggered, but no saved model found at {best_model_path_to_load} to load best state.")
            break
            
    artifacts['training_history'] = history
    # Save final artifacts. This will overwrite the artifacts_file_with_pkl potentially
    # with the latest history, even if the last epoch wasn't the best model.
    # The model file (.pth) remains the one from the best epoch.
    final_artifacts_path = Path(model_save_path_full).with_suffix('.pkl')
    with open(final_artifacts_path, 'wb') as f:
        pickle.dump(artifacts, f) # Save the complete artifacts dict with all history etc.
    logger.info(f"Final artifacts (including full history) saved to {final_artifacts_path}")

    logger.info("Training finished.")
    return model, artifacts

def load_pytorch_model(model_path_str: str):
    model_path = Path(model_path_str)
    artifacts_path = model_path.with_suffix('.pkl')

    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not artifacts_path.exists():
        logger.error(f"Artifacts file not found: {artifacts_path}")
        raise FileNotFoundError(f"Artifacts file not found: {artifacts_path}")

    logger.info(f"Loading model from {model_path} and artifacts from {artifacts_path}")
    with open(artifacts_path, 'rb') as f:
        artifacts = pickle.load(f)

    model_params = artifacts['model_params']
    model = TabularNNModel(
        embedding_sizes=model_params['embedding_sizes'],
        n_continuous=model_params['n_continuous'],
        n_classes=model_params['n_classes'],
        layers=model_params['layers'],
        p_dropout=model_params.get('p_dropout', 0.1) # provide default if not in older artifacts
    )
    model.load_state_dict(torch.load(model_path))
    model.eval() # Set to evaluation mode
    logger.info("Model and artifacts loaded successfully.")
    return model, artifacts

def preprocess_inference_data_pytorch(df_infer_pd: pd.DataFrame, artifacts: dict):
    logger.info("Preprocessing inference data...")
    df_processed = df_infer_pd.copy()

    categorical_feature_names = artifacts['categorical_feature_names']
    continuous_feature_names = artifacts['continuous_feature_names']
    
    # Handle NA values and encode categorical features
    for col in categorical_feature_names:
        if col not in df_processed.columns:
            logger.warning(f"Categorical column '{col}' not found in inference data. Skipping.")
            # Potentially add a column of NaNs or default values if strict schema adherence is needed
            # For now, we assume downstream model can handle missing embeddings if column is truly absent
            continue

        df_processed[col] = df_processed[col].astype(str)
        fill_val_from_artifact = artifacts['na_fills'].get(col, "MISSING_CAT") # Should be MISSING_CAT
        df_processed[col] = df_processed[col].fillna(fill_val_from_artifact)
        
        encoder = artifacts['categorical_encoders'][col]
        known_labels = set(encoder.classes_) # These are the labels seen during training + MISSING_CAT
        
        # Get the integer encoding of MISSING_CAT (or the fill_val_from_artifact)
        # This must exist if training saved it correctly.
        try:
            encoded_missing_val = encoder.transform([fill_val_from_artifact])[0]
        except ValueError:
            # This is a fallback if somehow fill_val_from_artifact is not in encoder.classes_
            # This indicates a problem with how artifacts were saved or if fill_val_from_artifact was corrupted.
            logger.error(f"Critical: Fill value '{fill_val_from_artifact}' for column '{col}' is not in its saved encoder's classes. Defaulting to encoding 0 for unknowns. This may lead to incorrect embeddings.")
            # Attempt to find a generic "MISSING_CAT" if the artifact one failed, else use 0.
            if "MISSING_CAT" in known_labels:
                encoded_missing_val = encoder.transform(["MISSING_CAT"])[0]
            else:
                encoded_missing_val = 0 # Fallback to 0, less ideal.

        # For each value, if it's known, transform it. If unknown, map to encoded_missing_val.
        transformed_col_values = []
        for x in df_processed[col]:
            if x in known_labels:
                transformed_col_values.append(encoder.transform([x])[0])
            else:
                # logger.warning(f"Unseen label '{x}' in column '{col}' during inference. Mapping to MISSING_CAT encoding ('{encoded_missing_val}').")
                transformed_col_values.append(encoded_missing_val)
        df_processed[col] = pd.Series(transformed_col_values, index=df_processed.index)

        # No need to check for -1 anymore, as all values should now be valid, non-negative indices.

    # Handle NA values and scale continuous features
    for col in continuous_feature_names:
        if col not in df_processed.columns:
            logger.warning(f"Continuous column '{col}' not found in inference data. Skipping.")
            continue
        fill_val_from_artifact = artifacts['na_fills'].get(col, "MISSING_CAT") # Should be MISSING_CAT
        df_processed[col] = df_processed[col].fillna(fill_val_from_artifact)
        df_processed[col] = artifacts['continuous_scalers'][col].transform(df_processed[col].values.reshape(-1, 1))

    X_cat_infer = torch.tensor(df_processed[categorical_feature_names].values, dtype=torch.long) if categorical_feature_names and all(c in df_processed.columns for c in categorical_feature_names) else torch.empty(len(df_processed), 0, dtype=torch.long)
    X_cont_infer = torch.tensor(df_processed[continuous_feature_names].values, dtype=torch.float32) if continuous_feature_names and all(c in df_processed.columns for c in continuous_feature_names) else torch.empty(len(df_processed), 0, dtype=torch.float32)
    
    # Handle cases where some features might be missing entirely from inference data after warnings
    # If X_cat_infer or X_cont_infer are not the expected shape based on training, model will fail.
    # This check ensures they are at least 2D.
    if categorical_feature_names and X_cat_infer.shape[1] != len(categorical_feature_names):
        logger.error(f"Mismatch in categorical feature count. Expected {len(categorical_feature_names)}, got {X_cat_infer.shape[1]}. This might be due to missing columns in inference data.")
        # Pad with a default value if necessary, or raise error. For now, this will likely cause error in model forward pass.
        # A robust solution would be to create columns of default encoded values (e.g., MISSING_CAT encoded)
        # if a categorical column was missing from df_infer_pd.

    if continuous_feature_names and X_cont_infer.shape[1] != len(continuous_feature_names):
        logger.error(f"Mismatch in continuous feature count. Expected {len(continuous_feature_names)}, got {X_cont_infer.shape[1]}.")
        # Pad with default scaled values (e.g., 0s if scaled around mean)

    logger.info("Inference data preprocessed.")
    return X_cat_infer, X_cont_infer

def get_pytorch_predictions(
    model: TabularNNModel,
    df_infer_input_pl: pl.DataFrame, # Renamed parameter for clarity
    artifacts: dict
):
    logger.info("Getting PyTorch predictions...")
    if not isinstance(df_infer_input_pl, pl.DataFrame):
        raise TypeError("Input df_infer_input_pl must be a Polars DataFrame.")

    # Convert input Polars DF to Pandas DF for processing & output structure
    df_infer_pd_full = df_infer_input_pl.to_pandas()
    results_df = df_infer_pd_full.copy() # This will be the base for our output

    target_name = artifacts.get('target_name', 'target') # Get target name from artifacts

    # Feature selection for preprocessing (similar to before, but on the pandas copy)
    req_cat_cols = artifacts.get('categorical_feature_names', [])
    req_cont_cols = artifacts.get('continuous_feature_names', [])
    all_req_model_features = req_cat_cols + req_cont_cols
    
    # Create a DataFrame for preprocessing containing only the features the model needs
    cols_for_preprocessing = []
    missing_model_features = []
    for col in all_req_model_features:
        if col in df_infer_pd_full.columns:
            cols_for_preprocessing.append(col)
        else:
            # This column is required by the model but not in input, will be handled by preprocess_inference_data_pytorch
            missing_model_features.append(col) 
            logger.warning(f"Model feature '{col}' not in inference input. Preprocessing will attempt to handle (e.g., add as NaN).")
            
    df_for_model_input = df_infer_pd_full[cols_for_preprocessing].copy()
    
    # preprocess_inference_data_pytorch expects all required columns to be present or it adds them as NaNs internally
    X_cat_infer, X_cont_infer = preprocess_inference_data_pytorch(df_for_model_input, artifacts)

    if X_cat_infer.nelement() == 0 and X_cont_infer.nelement() == 0 and (req_cat_cols or req_cont_cols):
        logger.warning("Both categorical and continuous tensors are empty for inference, but model expects features. Predictions might be non-sensical.")
    
    if X_cat_infer.shape[0] == 0 and X_cont_infer.shape[0] == 0 and len(df_infer_pd_full) > 0:
         logger.error("Preprocessed tensors have 0 rows but input dataframe had rows. This indicates an issue in preprocessing selection or data types.")
         # Add empty prediction columns to results_df to maintain structure, then return
         if target_name + "_Actual" not in results_df.columns and target_name in results_df.columns:
             results_df.rename(columns={target_name: target_name + "_Actual"}, inplace=True)
         results_df[target_name + '_Pred'] = pd.Series(dtype='object') # Changed to _Pred
         target_encoder = artifacts['target_encoder']
         class_labels = target_encoder.classes_
         for class_label in class_labels:
            results_df[f'prob_{class_label}'] = pd.Series(dtype='float')
         return results_df

    model.eval()
    with torch.no_grad():
        outputs = model(X_cat_infer, X_cont_infer)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_prob_values, predicted_indices = torch.max(outputs, 1) # Get both max probability and indices

    target_encoder = artifacts['target_encoder']
    predicted_labels_str = target_encoder.inverse_transform(predicted_indices.cpu().numpy())
    predicted_labels_code = predicted_indices.cpu().numpy()

    results_df[target_name + '_Pred'] = predicted_labels_str
    results_df[target_name + '_Pred_Code'] = predicted_labels_code

    # Handle Actual_Code and Targets_Code if original target is present
    if target_name in df_infer_pd_full.columns: # Check in the original full pandas df
        actual_labels_str_series = df_infer_pd_full[target_name].astype(str)
        try:
            # Ensure all actual labels are known to the encoder, fill with a placeholder if not, then transform
            # This step is crucial if df_test might contain target values not seen in training target_encoder fit
            known_target_classes = set(target_encoder.classes_)
            # Use a placeholder that is unlikely to be a real class, and ensure it's handled by encoder if possible
            # However, for safety, we should only transform known classes here.
            # If actual data has values not in encoder, an error would occur.
            # For now, assume actual_labels_str_series values are mostly covered by encoder.classes_
            # A more robust way would be to handle unknown actual labels if that's a scenario.
            actual_labels_code_series = target_encoder.transform(actual_labels_str_series)
            results_df[target_name + '_Actual_Code'] = actual_labels_code_series
            results_df[target_name + '_Targets_Code'] = actual_labels_code_series # Assuming Targets_Code is same as Actual_Code
        except ValueError as ve:
            logger.error(f"Error encoding actual target column '{target_name}' for '_Actual_Code': {ve}. Some actual values might not be in the target encoder. Filling code columns with NaN for this target.")
            results_df[target_name + '_Actual_Code'] = pd.NA
            results_df[target_name + '_Targets_Code'] = pd.NA
        
        # Rename the original string target column to _Actual (if not already done, though prior logic should handle it)
        if target_name in results_df.columns and target_name + "_Actual" not in results_df.columns:
             results_df.rename(columns={target_name: target_name + "_Actual"}, inplace=True)
        elif target_name + "_Actual" not in results_df.columns: # If original target_name column was not even there
             results_df[target_name + "_Actual"] = pd.NA # Add as NA if it was completely missing

    else:
        logger.info(f"Original target column '{target_name}' not in input. '_Actual', '_Actual_Code', '_Targets_Code' columns will be NaN or missing.")
        results_df[target_name + '_Actual'] = pd.NA
        results_df[target_name + '_Actual_Code'] = pd.NA
        results_df[target_name + '_Targets_Code'] = pd.NA

    # Add Match columns (handle potential NaNs from missing actuals/preds by ensuring columns exist)
    if (target_name + '_Actual' in results_df.columns and 
        target_name + '_Pred' in results_df.columns):
        results_df[target_name + '_Match'] = (results_df[target_name + '_Actual'] == results_df[target_name + '_Pred'])
    else:
        results_df[target_name + '_Match'] = False # Or pd.NA if preferred for missing data
        
    if (target_name + '_Actual_Code' in results_df.columns and 
        target_name + '_Pred_Code' in results_df.columns):
        # Ensure they are of compatible types for comparison if one could be pd.NA
        actual_codes_for_match = pd.to_numeric(results_df[target_name + '_Actual_Code'], errors='coerce')
        pred_codes_for_match = pd.to_numeric(results_df[target_name + '_Pred_Code'], errors='coerce')
        results_df[target_name + '_Match_Code'] = (actual_codes_for_match == pred_codes_for_match)
    else:
        results_df[target_name + '_Match_Code'] = False # Or pd.NA if preferred for missing data

    # Create a DataFrame for all probability columns at once
    class_labels = target_encoder.classes_
    prob_df_data = {}
    # Ensure probabilities tensor has the correct shape (batch_size, num_classes)
    if probabilities.ndim == 2 and probabilities.shape[0] == len(results_df) and probabilities.shape[1] == len(class_labels):
        for i, class_label in enumerate(class_labels):
            prob_df_data[f'prob_{class_label}'] = probabilities[:, i].cpu().numpy()
        prob_df = pd.DataFrame(prob_df_data, index=results_df.index) 
        # Concatenate the probability DataFrame with the main results DataFrame
        results_df = pd.concat([results_df, prob_df], axis=1)
    else:
        logger.warning(f"Probabilities tensor shape mismatch or empty. Expected ({len(results_df)}, {len(class_labels)}), got {probabilities.shape}. Skipping probability columns.")
        # Optionally, add empty prob columns to maintain schema consistency if needed
        for class_label in class_labels:
            results_df[f'prob_{class_label}'] = np.nan # Or pd.NA

    logger.info(f"Final columns in results_df before returning: {results_df.columns.tolist()}")
    logger.info(f"Shape of results_df before returning: {results_df.shape}")
    logger.info("Predictions generated and merged with input data.")
    return results_df

if __name__ == '__main__':
    # Example Usage (Illustrative - requires actual data and setup)
    logger.info("Running example usage of mlBridgePytorchLib_gemini25pro.py (illustrative)")

    # 0. Setup paths and parameters
    SAVED_MODELS_PATH = Path("./saved_models_pytorch_test")
    SAVED_MODELS_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = "example_pytorch_model"
    MODEL_SAVE_PATH_FULL = str(SAVED_MODELS_PATH / MODEL_NAME) # Base path, e.g., "./saved_models_pytorch_test/example_pytorch_model"

    # 1. Create dummy Polars DataFrame for training
    n_rows = 200
    data = {
        'cat_feat1': [f'A{i%5}' for i in range(n_rows)],
        'cat_feat2': [f'B{i%3}' for i in range(n_rows)],
        'cont_feat1': np.random.rand(n_rows) * 10,
        'cont_feat2': np.random.randn(n_rows) * 5,
        'target': [f'Class_{i%2}' for i in range(n_rows)] # Binary classification example
    }
    df_pl = pl.DataFrame(data)

    # Define game_state_columns (can be more sophisticated)
    game_state_cols_example = {
        'categorical': ['cat_feat1', 'cat_feat2'],
        'continuous': ['cont_feat1', 'cont_feat2'],
        # 'target_col': 'target' # Target is passed separately
    }
    
    logger.info(f"Dummy Polars DataFrame created with shape: {df_pl.shape}")
    logger.info(f"Columns: {df_pl.columns}")
    logger.info(f"Target distribution:\n{df_pl['target'].value_counts()}")


    # 2. Train the model
    try:
        trained_model, training_artifacts = train_pytorch_model(
            df_train_full=df_pl,
            target_name='target',
            game_state_columns=game_state_cols_example,
            model_save_path_full=MODEL_SAVE_PATH_FULL, # .pth added automatically for model, .pkl for artifacts
            nsamples=-1, # Use all data
            epochs=5, # Short training for example
            batch_size=32,
            early_stopping_patience=2,
            lr=0.01,
            layers_config=[50, 25],
            p_dropout=0.2
        )
        logger.info(f"Training completed. Model type: {type(trained_model)}")
        logger.info(f"Artifacts keys: {training_artifacts.keys()}")
        
        # Check if model file and artifact file were created
        # MODEL_SAVE_PATH_FULL is "path/to/basename"
        model_file = Path(MODEL_SAVE_PATH_FULL).with_suffix(".pth") 
        artifact_file = Path(MODEL_SAVE_PATH_FULL).with_suffix(".pkl") 

        if model_file.exists():
            logger.info(f"Model file saved: {model_file}")
        else:
            logger.error(f"Model file NOT saved: {model_file}")

        if artifact_file.exists():
            logger.info(f"Artifact file saved: {artifact_file}")
        else:
            logger.error(f"Artifact file NOT saved: {artifact_file}")


    except Exception as e:
        logger.error(f"Error during training example: {e}", exc_info=True)
        trained_model, training_artifacts = None, None # Ensure they are defined

    # 3. Load the model (if training was successful)
    if trained_model: # Or check if model_file exists
        try:
            # Construct the full path to the .pth model file for loading
            path_to_pth_model_file = str(Path(MODEL_SAVE_PATH_FULL).with_suffix(".pth"))
            loaded_model, loaded_artifacts = load_pytorch_model(path_to_pth_model_file)
            logger.info(f"Model loaded successfully. Type: {type(loaded_model)}")
            logger.info(f"Loaded artifacts keys: {loaded_artifacts.keys()}")

            # 4. Create dummy Polars DataFrame for inference
            n_infer_rows = 50
            infer_data = {
                'cat_feat1': [f'A{i%6}' for i in range(n_infer_rows)], # Includes a new category A5
                'cat_feat2': [None] * int(n_infer_rows/5) + [f'B{i%3}' for i in range(n_infer_rows - int(n_infer_rows/5))], # Includes NaNs
                'cont_feat1': np.random.rand(n_infer_rows) * 12, # Slightly different range
                'cont_feat2': np.random.randn(n_infer_rows) * 6,
                'extra_col': ['foo'] * n_infer_rows # Extra column to be ignored
            }
            # Missing 'target' column, as expected for inference data
            df_infer_pl = pl.DataFrame(infer_data)
            logger.info(f"Dummy inference Polars DataFrame created with shape: {df_infer_pl.shape}")
            logger.info(f"Inference columns: {df_infer_pl.columns}")


            # 5. Get predictions
            predictions_df = get_pytorch_predictions(loaded_model, df_infer_pl, loaded_artifacts)
            logger.info("Predictions DataFrame:")
            print(predictions_df.head())
            logger.info(f"Predictions shape: {predictions_df.shape}")
            if not predictions_df.empty:
                 logger.info(f"Predicted target distribution:\n{predictions_df[loaded_artifacts['target_name'] + '_predicted'].value_counts()}")

            # Test with an empty dataframe
            logger.info("Testing with empty inference DataFrame:")
            empty_df_infer_pl = pl.DataFrame({col: pl.Series([], dtype=df_pl[col].dtype) for col in df_infer_pl.columns})
            empty_predictions_df = get_pytorch_predictions(loaded_model, empty_df_infer_pl, loaded_artifacts)
            logger.info(f"Empty predictions DataFrame shape: {empty_predictions_df.shape}")
            print(empty_predictions_df)
            
            # Test with dataframe having only some columns
            logger.info("Testing with inference DataFrame missing some features:")
            partial_infer_data = {'cat_feat1': [f'A{i%2}' for i in range(5)]} # Only one feature
            df_partial_infer_pl = pl.DataFrame(partial_infer_data)
            partial_predictions_df = get_pytorch_predictions(loaded_model, df_partial_infer_pl, loaded_artifacts)
            logger.info("Partial predictions DataFrame:")
            print(partial_predictions_df.head())


        except Exception as e:
            logger.error(f"Error during loading/prediction example: {e}", exc_info=True)
    else:
        logger.warning("Skipping load/prediction example as training might have failed.")

    logger.info("Example usage finished.") 