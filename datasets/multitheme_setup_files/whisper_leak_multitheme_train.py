#!/usr/bin/env python3
from pathlib import Path
from core.classifiers.base_classifier import BaseClassifier
from core.classifiers.cnn_classifier import CNNClassifier
from core.classifiers.attention_bi_lstm_classifier import AttentionBiLSTMClassifier
from core.classifiers.bert_time_series_classifier import BERTTimeSeriesClassifier
from core.classifiers.combined_lstm_bert_classifier import CombinedLSTMBERTClassifier
from core.classifiers.lightgbm_classifier import LightGBMClassifier
from core.classifiers.knn_classifier import KNNClassifier
from core.classifiers.rnn_classifier import RNNClassifier
from core.classifiers.utils import (
    EarlyStopping, ModelTrainer, load_chatbot_data,
    set_seed, split_data
)
from core.classifiers.loader import Loader

from core.classifiers.visualization import calculate_metrics, set_plot_style
from core.classifiers.visualization import plot_training_curves
from core.classifiers.visualization import plot_roc_curve
from core.classifiers.visualization import plot_precision_recall_curve
from core.classifiers.visualization import plot_confusion_matrix 
from core.classifiers.visualization import plot_score_distribution
from core.classifiers.visualization import create_model_dashboard

from core.utils import ThrowingArgparse
from core.utils import PrintUtils
from core.utils import OsUtils
from core.chatbot_utils import ChatbotUtils

import json
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Multi-theme data loading
# ---------------------------------------------------------------------------

def load_multitheme_data(chatbot, input_folder, prompts_path, downsample=1.0,
                         multitheme_mode='binary', neutral_theme='other'):
    """
    Load and label data from a multi-theme prompts JSON file.

    The JSON format expected is:
        {
            "<theme>": {"repeat": <int>, "prompts": ["prompt1", ...]},
            ...
        }

    Args:
        chatbot: Chatbot name used to locate raw timing data via load_chatbot_data.
        input_folder: Input folder passed through to load_chatbot_data.
        prompts_path: Path to the multi-theme JSON file.
        downsample: Fraction of data to keep (passed through).
        multitheme_mode: 'binary'     → neutral_theme gets label 0, all others get 1.
                         'multiclass' → each theme gets its own integer label.
        neutral_theme: The theme key treated as the negative/neutral class in
                       binary mode (default: 'other').

    Returns:
        df: DataFrame with an updated 'target' column (int).
        label_map: dict mapping theme name → integer label.
                   In binary mode: {neutral_theme: 0, 'sensitive': 1}.
                   In multiclass mode: {theme: idx, ...} sorted alphabetically.
    """
    PrintUtils.start_stage(f'Loading multi-theme prompts from {prompts_path}')
    with open(prompts_path, 'r', encoding='utf-8') as fh:
        theme_data = json.load(fh)

    themes = list(theme_data.keys())
    PrintUtils.print_extra(f'Themes found: {", ".join(themes)}')

    # Build label map
    if multitheme_mode == 'binary':
        if neutral_theme not in themes:
            raise ValueError(
                f"neutral_theme '{neutral_theme}' not found in JSON keys: {themes}"
            )
        label_map = {neutral_theme: 0}
        for t in themes:
            if t != neutral_theme:
                label_map[t] = 1
        PrintUtils.print_extra(
            f'Binary mode — neutral class: "{neutral_theme}" (0), '
            f'sensitive themes: {[t for t in themes if t != neutral_theme]} (1)'
        )
    elif multitheme_mode == 'multiclass':
        sorted_themes = sorted(themes)
        label_map = {t: i for i, t in enumerate(sorted_themes)}
        PrintUtils.print_extra(
            f'Multiclass mode — label map: {label_map}'
        )
    else:
        raise ValueError(
            f"multitheme_mode must be 'binary' or 'multiclass', got '{multitheme_mode}'"
        )

    # Build a flat prompts → label mapping so we can re-label the existing df
    # after load_chatbot_data fills it in.
    prompt_to_label = {}
    prompt_to_theme = {}
    for theme, content in theme_data.items():
        lbl = label_map[theme]
        for p in content['prompts']:
            prompt_to_label[p] = lbl
            prompt_to_theme[p] = theme

    PrintUtils.print_extra(
        f'Total unique prompts across all themes: {len(prompt_to_label)}'
    )
    PrintUtils.end_stage()

    # Load raw timing data using the existing utility
    # (it will assign binary targets based on the JSON; we overwrite them)
    df = load_chatbot_data(chatbot, input_folder, prompts_path, downsample)

    # Overwrite 'target' with multi-theme labels
    PrintUtils.start_stage('Re-labelling data with multi-theme targets')
    df['theme'] = df['prompt'].map(prompt_to_theme)
    df['target'] = df['prompt'].map(prompt_to_label)

    unknown_mask = df['target'].isna()
    if unknown_mask.any():
        n_unknown = unknown_mask.sum()
        PrintUtils.print_extra(
            f'Warning: {n_unknown} rows have prompts not found in the JSON '
            f'and will be dropped.'
        )
        df = df[~unknown_mask].copy()

    df['target'] = df['target'].astype(int)
    PrintUtils.print_extra(f'Label distribution:\n{df["target"].value_counts().sort_index().to_string()}')
    PrintUtils.end_stage()

    return df, label_map


def is_multitheme_json(prompts_path):
    """
    Heuristic: returns True if the JSON looks like a multi-theme file
    (top-level dict whose values have 'prompts' and 'repeat' keys),
    False if it looks like the legacy binary format.
    """
    try:
        with open(prompts_path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return False
        first_val = next(iter(data.values()))
        return isinstance(first_val, dict) and 'prompts' in first_val
    except Exception:
        return False


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Namespace: Parsed arguments
    """
    # Load all chatbots
    PrintUtils.start_stage('Loading chatbots')
    chatbots = ChatbotUtils.load_chatbots(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chatbots')
    )
    assert len(chatbots) > 0, Exception('Could not load any chatbots')
    chatbot_names = ', '.join([f'*{chatbot.__name__}*' for chatbot in chatbots.values()])
    PrintUtils.print_extra(f'Loaded chatbots: {chatbot_names}')
    PrintUtils.end_stage()

    # Parsing arguments
    PrintUtils.start_stage('Parsing command-line arguments')
    parser = ThrowingArgparse()
    parser.add_argument('-c', '--chatbot', help='The chatbot.', required=True)
    parser.add_argument('-m', '--modeltype', help='The model type (CNN, LSTM, LSTM_BERT, LGBM, KNN, or BERT).', default='CNN')
    parser.add_argument('-p', '--prompts', help='The prompts JSON file path', default='./prompts/standard/prompts.json')
    parser.add_argument('-s', '--seed', type=int, help='The random seed', default=42)
    parser.add_argument('-b', '--batchsize', type=int, help='The batch size', default=32)
    parser.add_argument('-e', '--epochs', type=int, help='The number of epochs', default=200)
    parser.add_argument('-P', '--patience', type=int, help='The patience value', default=5)
    parser.add_argument('-k', '--kernelwidth', type=int, help='The kernel width', default=3)
    parser.add_argument('-l', '--learningrate', type=float, help='The learning rate', default=0.0001)
    parser.add_argument('-t', '--testsize', type=int, help='The test size in percentage', default=20)
    parser.add_argument('-v', '--validsize', type=int, help='The validation size in percentage taken from train set', default=5)
    parser.add_argument('-ds', '--downsample', type=float, help='Downsample the dataset', default=1.0)
    parser.add_argument('-i', '--input_folder', type=str, help='Input folder for the data', default='data_v2')
    parser.add_argument('-C', '--csv_output_only', action='store_true', help='Only output train/valid/test CSV files without training models', default=False)
    # --- Multi-theme arguments ---
    parser.add_argument(
        '--multitheme_mode',
        choices=['binary', 'multiclass'],
        default='binary',
        help=(
            'How to handle multi-theme JSON files. '
            '"binary": neutral_theme → 0, all other themes → 1 (default). '
            '"multiclass": each theme gets its own integer label.'
        )
    )
    parser.add_argument(
        '--neutral_theme',
        type=str,
        default='other',
        help='Theme key treated as the negative/neutral class in binary mode (default: "other").'
    )
    args = parser.parse_args()
    
    # Validate arguments
    assert args.seed >= 0, Exception(f'Invalid random seed: {args.seed}')
    assert args.batchsize > 0, Exception(f'Invalid batch size: {args.batchsize}')
    assert args.epochs > 0, Exception(f'Invalid number of epochs: {args.epochs}')
    assert args.patience > 0, Exception(f'Invalid patience value: {args.patience}')
    assert args.kernelwidth > 0, Exception(f'Invalid kernel width: {args.kernelwidth}')
    assert 0 < args.learningrate < 1, Exception(f'Invalid learning rate: {args.learningrate}')
    assert 0 < args.testsize < 100, Exception(f'Invalid test size percentage: {args.testsize}')
    assert 0 < args.validsize < 100, Exception(f'Invalid validation size percentage: {args.validsize}')
    assert len(args.chatbot) > 0, Exception('Chatbot name cannot be empty')
    if args.modeltype not in ('CNN', 'LSTM', 'LSTM_BERT', 'BERT', 'LGBM', 'KNN', 'RNN'):
        raise Exception(f'Invalid model type: {args.modeltype}')
    PrintUtils.end_stage()

    return args

def make_flat_features(df, max_len):
    rows = []
    for _, r in df.iterrows():
        dl = np.array(r['data_lengths'], dtype=np.float32)
        td = np.array(r['time_diffs'], dtype=np.float32)
        # Truncate or pad each sequence to max_len
        max_len=min(max_len, 30)
        dl = dl[:max_len] if len(dl) >= max_len else np.pad(dl, (0, max_len - len(dl)))
        td = td[:max_len] if len(td) >= max_len else np.pad(td, (0, max_len - len(td)))
        rows.append(np.concatenate([dl, td]))
    return np.array(rows, dtype=np.float32)

def process_data_for_csv_export(data_dir, df_train, df_val, df_test, train_dataset, val_dataset, test_dataset):
    """
    Process and save data in CSV format for both normalized and non-normalized versions.
    
    Args:
        data_dir: Directory to save CSVs to
        df_train, df_val, df_test: Original, non-normalized dataframes
        train_dataset, val_dataset, test_dataset: Normalized dataset objects
    """
    # Save the train, validation, and test sets to CSV files (original format)
    train_csv_path = os.path.join(data_dir, 'train_original_format.csv')
    val_csv_path = os.path.join(data_dir, 'val_original_format.csv')
    test_csv_path = os.path.join(data_dir, 'test_original_format.csv')

    train_dataset.df.to_csv(train_csv_path, index=False)
    val_dataset.df.to_csv(val_csv_path, index=False)
    test_dataset.df.to_csv(test_csv_path, index=False)

    PrintUtils.print_extra(f'Original format train set saved to *{os.path.basename(train_csv_path)}*')
    PrintUtils.print_extra(f'Original format validation set saved to *{os.path.basename(val_csv_path)}*')
    PrintUtils.print_extra(f'Original format test set saved to *{os.path.basename(test_csv_path)}*')

    # Get the max_len determined by the training set
    max_len = train_dataset.max_len
    PrintUtils.print_extra(f'Using max_len = {max_len} for expanding columns.')

    # Process and save normalized data
    PrintUtils.print_extra("Processing normalized data...")
    expand_and_save_df(train_dataset.df, max_len, 'train', data_dir, True)
    expand_and_save_df(val_dataset.df, max_len, 'val', data_dir, True)
    expand_and_save_df(test_dataset.df, max_len, 'test', data_dir, True)

    # Process and save non-normalized data
    PrintUtils.print_extra("Processing non-normalized data...")
    expand_and_save_df(df_train, max_len, 'train', data_dir, False)
    expand_and_save_df(df_val, max_len, 'val', data_dir, False)
    expand_and_save_df(df_test, max_len, 'test', data_dir, False)

def expand_and_save_df(df, max_len, base_filename, results_dir, is_normalized):
    """
    Expands sequence columns and saves the DataFrame to CSV.
    
    Args:
        df: DataFrame to process
        max_len: Maximum sequence length
        base_filename: Base name for output file
        results_dir: Directory to save to
        is_normalized: Whether the data is normalized
    """
    # Padding value (use 0.0 for consistency, especially after normalization)
    padding_value = 0.0

    # Process 'data_lengths'
    data_lengths_processed = df['data_lengths'].apply(
        lambda x: list(x[:max_len]) + [padding_value] * (max_len - len(x)) if len(x) < max_len else list(x[:max_len])
    ).tolist()
    data_lengths_cols = [f'data_length_{i}' for i in range(max_len)]
    df_data_lengths = pd.DataFrame(data_lengths_processed, columns=data_lengths_cols, index=df.index)

    # Process 'time_diffs'
    time_diffs_processed = df['time_diffs'].apply(
        lambda x: list(x[:max_len]) + [padding_value] * (max_len - len(x)) if len(x) < max_len else list(x[:max_len])
    ).tolist()
    time_diffs_cols = [f'time_diff_{i}' for i in range(max_len)]
    df_time_diffs = pd.DataFrame(time_diffs_processed, columns=time_diffs_cols, index=df.index)

    # Combine target and expanded columns
    if 'target' not in df.columns:
        PrintUtils.print_extra(f"Warning: 'target' column not found in DataFrame for {base_filename}. Skipping target column.")
        expanded_df = pd.concat([df_data_lengths, df_time_diffs], axis=1)
    else:
        expanded_df = pd.concat([df[['target']], df_data_lengths, df_time_diffs], axis=1)

    # Construct filename and save
    norm_suffix = 'normalized' if is_normalized else 'non_normalized'
    filename = f"{base_filename}_expanded_{norm_suffix}.csv"
    filepath = os.path.join(results_dir, filename)
    expanded_df.to_csv(filepath, index=False)
    PrintUtils.print_extra(f'Saved expanded data to *{filename}*')

def create_model(model_type, norm, kernel_width=3, df_train=None, num_classes=2):
    """
    Create and return the specified model type with given parameters.
    
    Args:
        model_type: The type of model to create (CNN, LSTM, LSTM_BERT, BERT)
        norm: Normalization parameters
        kernel_width: Kernel width for CNN models
        df_train: Training dataframe (required for BERT models)
        num_classes: Number of output classes (2 for binary, >2 for multiclass)
        
    Returns:
        model: The created model
        model_path: Path where the model should be saved
    """
    models_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'models'
    )
    
    # Suffix to distinguish binary vs multiclass saved models
    class_suffix = 'binary' if num_classes == 2 else f'{num_classes}class'

    model_type = model_type.upper()
    if model_type == 'CNN':
        model = CNNClassifier(norm, kernel_width, num_classes=num_classes)
        model_path = os.path.join(models_dir, f'cnn_{class_suffix}_classifier.pth')
    elif model_type == 'LSTM':
        model = AttentionBiLSTMClassifier(norm, num_classes=num_classes)
        model_path = os.path.join(models_dir, f'lstm_{class_suffix}_classifier.pth')
    elif model_type == 'LSTM_BERT':
        model = CombinedLSTMBERTClassifier(norm, num_classes=num_classes)
        model_path = os.path.join(models_dir, f'lstm_bert_{class_suffix}_classifier.pth')
    elif model_type == 'LGBM':
        model = LightGBMClassifier(norm, num_classes=num_classes)
        model_path = os.path.join(models_dir, f'lightgbm_{class_suffix}_classifier.pth')
    elif model_type == "BERT":
        # Calculate the token boundary parameters
        (time_boundaries_norm, len_boundaries_norm) = BERTTimeSeriesClassifier.calculate_boundaries(
            df_train,
            num_buckets=50,
            norm=norm
        )

        model = BERTTimeSeriesClassifier(
            norm,
            time_boundaries_norm,
            len_boundaries_norm,
            num_buckets=50,
            num_classes=num_classes,
        )
        model_path = os.path.join(models_dir, f'bert_{class_suffix}_classifier.pth')
    elif model_type == "KNN":
        model = KNNClassifier(norm, n_neighbors=5, num_classes=num_classes)
        model_path = os.path.join(models_dir, f'knn_{class_suffix}_classifier.pth')
    elif model_type == "RNN":
        model = RNNClassifier(norm, num_classes=num_classes)
        model_path = os.path.join(models_dir, f'rnn_{class_suffix}_classifier.pth')
    else:
        raise ValueError(f'Unsupported model type: {model_type}')
    
    return model, model_path

def main():
    """
    Main routine.
    """
    # Catch-all for clean error handling
    is_user_cancelled = False
    last_error = None
    
    try:
        # Print logo
        PrintUtils.print_logo()

        # Parse arguments
        args = parse_arguments()

        PrintUtils.start_stage("Ensuring data directory exists")
        training_data_dir = Path(__file__).parent.parent / "data"
        training_data_dir.mkdir(parents=True, exist_ok=True)
        PrintUtils.end_stage()

        # Setup
        set_plot_style()
        set_seed(args.seed)
        
        # Create directories
        PrintUtils.start_stage('Making directories')
        models_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'models'
        )
        results_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'results'
        )
        
        OsUtils.mkdir(models_dir)
        OsUtils.mkdir(results_dir)
        PrintUtils.end_stage()
   
        # Set device
        PrintUtils.start_stage('Setting device')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        PrintUtils.end_stage()
        PrintUtils.print_extra(f'Using device: *{device}*')

        # Load data — auto-detect multi-theme JSON or fall back to legacy format
        if is_multitheme_json(args.prompts):
            PrintUtils.print_extra(
                f'Detected multi-theme JSON format '
                f'(mode: *{args.multitheme_mode}*, neutral theme: *{args.neutral_theme}*)'
            )
            df, label_map = load_multitheme_data(
                args.chatbot,
                args.input_folder,
                args.prompts,
                args.downsample,
                multitheme_mode=args.multitheme_mode,
                neutral_theme=args.neutral_theme,
            )
            num_classes = len(set(label_map.values()))
            PrintUtils.print_extra(f'Number of classes: *{num_classes}*  |  Label map: {label_map}')
        else:
            PrintUtils.print_extra('Detected legacy binary JSON format')
            df = load_chatbot_data(args.chatbot, args.input_folder, args.prompts, args.downsample)
            num_classes = 2
            label_map = {'negative': 0, 'positive': 1}

        # Split data
        PrintUtils.start_stage('Splitting data into train, validation, and test sets')
        df_train, df_val, df_test = split_data(df, args.seed, args.testsize / 100.0, args.validsize / 100.0)
        PrintUtils.print_extra(f'Train set size: {len(df_train)}')
        PrintUtils.print_extra(f'Validation set size: {len(df_val)}')
        PrintUtils.print_extra(f'Test set size: {len(df_test)}')
        PrintUtils.end_stage()

        # Print detailed statistics about the data
        PrintUtils.print_extra(f'Unique prompts in train set: {len(df_train["prompt"].unique())}')
        PrintUtils.print_extra(f'Unique prompts in validation set: {len(df_val["prompt"].unique())}')
        PrintUtils.print_extra(f'Unique prompts in test set: {len(df_test["prompt"].unique())}')
        PrintUtils.print_extra(f'Unique prompts in train set by label: {df_train.groupby("target")["prompt"].nunique().to_dict()}')
        PrintUtils.print_extra(f'Unique prompts in validation set by label: {df_val.groupby("target")["prompt"].nunique().to_dict()}')
        PrintUtils.print_extra(f'Unique prompts in test set by label: {df_test.groupby("target")["prompt"].nunique().to_dict()}')
        PrintUtils.print_extra(f'Count of prompts in train set by label: {df_train.groupby("target")["prompt"].count().to_dict()}')
        PrintUtils.print_extra(f'Count of prompts in validation set by label: {df_val.groupby("target")["prompt"].count().to_dict()}')
        PrintUtils.print_extra(f'Count of prompts in test set by label: {df_test.groupby("target")["prompt"].count().to_dict()}')
        PrintUtils.print_extra(f'Average sequence length in train set: {df_train["data_lengths"].apply(len).mean()}')
        PrintUtils.print_extra(f'Average sequence length in validation set: {df_val["data_lengths"].apply(len).mean()}')
        PrintUtils.print_extra(f'Average sequence length in test set: {df_test["data_lengths"].apply(len).mean()}')

        # Prepare data
        PrintUtils.start_stage('Preparing data')
        train_dataset = Loader(df_train)
        val_dataset = Loader(df_val)
        test_dataset = Loader(df_test)

        norm = train_dataset.get_normalization()
        train_dataset.apply_normalization(norm)
        val_dataset.apply_normalization(norm)
        test_dataset.apply_normalization(norm)

        PrintUtils.end_stage()
        
        PrintUtils.print_extra(f'Max sequence length being used for model (95th percentile): *{train_dataset.max_len}*')

        # If only CSV export is requested, do that and exit
        if args.csv_output_only:
            process_data_for_csv_export(
                results_dir, 
                df_train, df_val, df_test, 
                train_dataset, val_dataset, test_dataset
            )
            return
    
        # Create model
        PrintUtils.start_stage('Instantiating model')
        model, model_path = create_model(
            args.modeltype, 
            norm, 
            args.kernelwidth, 
            df_train,
            num_classes=num_classes,
        )
        
        PrintUtils.print_extra(f'Model created: *{model.__class__.__name__}*')
        PrintUtils.end_stage()
        
        # Configure training parameters
        config = type('Config', (), {
            'max_epochs': args.epochs,
            'learning_rate': args.learningrate,
            'patience': args.patience,
            'batch_size': args.batchsize
        })
        
        # Create trainer and train model
        if args.modeltype == 'KNN':
            PrintUtils.start_stage('Training KNN model')
            max_len = train_dataset.max_len
            X_train = make_flat_features(train_dataset.df, max_len)
            y_train = train_dataset.df['target'].values
            model.fit(X_train, y_train)
            history = {'best_epoch': 1, 'train_losses': [], 'val_losses': [], 'train_accs': [], 'val_accs': []}
            PrintUtils.end_stage()
        else:
            checkpoint_path = os.path.join(models_dir, 'checkpoint.pt')

            model_trainer = ModelTrainer(model, config, device)
            history = model_trainer.fit(
                train_dataset,
                val_dataset,
                checkpoint_path,
                args.batchsize
            )
            checkpoint_path = os.path.join(models_dir, 'checkpoint.pt')

        best_epoch = history.get('best_epoch', 0)
        train_losses = history.get('train_losses', [])
        val_losses = history.get('val_losses', [])
        train_accs = history.get('train_accs', [])
        val_accs = history.get('val_accs', [])
        
        PrintUtils.print_extra(f'Best model found at epoch *{best_epoch}*')
        PrintUtils.end_stage()
        
        # Save model
        PrintUtils.start_stage('Saving model')
        model.save(model_path)
        PrintUtils.end_stage()
        
        # Evaluate on test set
        PrintUtils.start_stage('Inferencing on test dataset and generating metrics')

        is_multiclass = num_classes > 2

        if args.modeltype == 'KNN':
            X_test = make_flat_features(test_dataset.df, max_len)
            if is_multiclass:
                # predict_proba returns (N, num_classes); keep all columns
                test_scores_all = model.predict_proba(X_test)
                test_labels = test_dataset.df['target'].values
                test_preds = np.argmax(test_scores_all, axis=1)
                # For ROC/PR curves (OvR) and distribution we use the max prob
                test_scores = test_scores_all.max(axis=1)
            else:
                test_scores = model.predict_proba(X_test)[:, 1]
                test_labels = test_dataset.df['target'].values
                test_preds = (test_scores > 0.5).astype(int)
        else:
            if is_multiclass:
                # model_trainer.predict should return raw logits or softmax probs
                test_scores_all, test_labels, _ = model_trainer.predict(
                    test_dataset, batch_size=args.batchsize
                )
                # test_scores_all shape: (N, num_classes) or (N,) if model returns argmax
                if test_scores_all.ndim == 2:
                    test_preds = np.argmax(test_scores_all, axis=1)
                    test_scores = test_scores_all.max(axis=1)
                else:
                    # Already argmax — keep as-is, derive dummy scores
                    test_preds = test_scores_all.astype(int)
                    test_scores = test_scores_all.astype(float)
            else:
                test_scores, test_labels, _ = model_trainer.predict(
                    test_dataset, batch_size=args.batchsize
                )
                test_preds = (test_scores > 0.5).astype(int)
        # Generate visualizations
        plot_training_curves(
            train_losses, val_losses, 
            train_accs, val_accs, 
            best_epoch, 
            os.path.join(results_dir, 'training_curves.png')
        )

        if not is_multiclass:
            # These plots are designed for binary classification only
            plot_roc_curve(
                test_labels, test_scores, 
                os.path.join(results_dir, 'roc_curve.png')
            )
            
            plot_precision_recall_curve(
                test_labels, test_scores, 
                os.path.join(results_dir, 'precision_recall_curve.png')
            )

            create_model_dashboard(
                test_scores, test_labels, 
                train_losses, val_losses, 
                best_epoch,  
                os.path.join(results_dir, 'model_performance_dashboard.png')
            )
            
            plot_score_distribution(
                test_scores, test_labels, 
                os.path.join(results_dir, 'prediction_score_distribution.png')
            )
        else:
            PrintUtils.print_extra(
                'Skipping binary-only plots (ROC, PR curve, score distribution) '
                'in multiclass mode.'
            )
        
        conf_matrix = plot_confusion_matrix(
            test_labels, test_preds, 
            os.path.join(results_dir, 'confusion_matrix.png'),
            **(
                {'labels': sorted(label_map.values()),
                 'display_labels': [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]}
                if is_multiclass else {}
            )
        )
        
        # Save test predictions
        df_test_results = test_dataset.df.copy()
        df_test_results['prediction'] = test_preds
        df_test_results['score'] = test_scores
        # In multitheme mode, add human-readable theme and predicted theme columns
        if is_multiclass and 'theme' in df.columns:
            inv_label_map = {v: k for k, v in label_map.items()}
            df_test_results['theme'] = df_test.get('theme', pd.Series(dtype=str))
            df_test_results['predicted_theme'] = [inv_label_map.get(p, str(p)) for p in test_preds]
        df_test_results.to_csv(os.path.join(results_dir, 'test_results.csv'), index=False)
        PrintUtils.end_stage()
        PrintUtils.print_extra('Results saved to *test_results.csv*')
        
        # Print confusion matrix metrics
        PrintUtils.start_stage('Printing confusion matrix metrics')
        if is_multiclass:
            # calculate_metrics is designed for binary; compute basic multiclass metrics here
            from sklearn.metrics import classification_report, accuracy_score
            acc = accuracy_score(test_labels, test_preds)
            report = classification_report(
                test_labels, test_preds,
                target_names=[k for k, _ in sorted(label_map.items(), key=lambda x: x[1])],
                output_dict=True,
            )
            metrics = {'accuracy': round(acc, 4)}
            for class_name, class_metrics in report.items():
                if isinstance(class_metrics, dict):
                    for metric_name, val in class_metrics.items():
                        metrics[f'{class_name}_{metric_name}'] = round(val, 4)
        else:
            metrics = calculate_metrics(test_labels, test_scores, test_preds, conf_matrix, df_test)

        # Display metrics
        for key, value in metrics.items():
            PrintUtils.print_extra(f'{key}: {value}')

        # Write metrics to file
        with open(os.path.join(results_dir, 'confusion_matrix_metrics.txt'), 'w') as f:
            for key, value in metrics.items():
                f.write(f'{key}: {value}\n')
            
        PrintUtils.print_extra(f'Metrics saved to *confusion_matrix_metrics.txt*')
        PrintUtils.end_stage()
        
        # Test the inference function
        PrintUtils.start_stage('Testing inference function')
        loaded_model = BaseClassifier.load(model_path, device)

        # Test with tuple input
        sample_row = df_test.iloc[0]
        time_diffs, data_lengths = sample_row['time_diffs'], sample_row['data_lengths']
        prob, pred = loaded_model.inference((time_diffs, data_lengths), device)
        PrintUtils.print_extra(f'Inference result (tuple input): prob=*{prob:.4f}*, pred=*{pred}*')
        
        # Test with DataFrame input
        sample_df = df_test.iloc[[0]]
        probs, preds = loaded_model.inference(sample_df, device)

        prob_arr = np.asarray(probs)
        pred_arr = np.asarray(preds)

        prob_val = prob_arr.reshape(-1).item()
        pred_val = pred_arr.reshape(-1).item()

        PrintUtils.print_extra(
            f'Inference result (DataFrame input): prob=*{prob_val:.4f}*, pred=*{int(pred_val)}*'
        )
        #probs, preds = loaded_model.inference(sample_df, device)
        #PrintUtils.print_extra(f'Inference result (DataFrame input): prob=*{probs[0]:.4f}*, pred=*{preds[0]}*')
        #PrintUtils.end_stage()

    except KeyboardInterrupt:
        if PrintUtils.is_in_stage():
            PrintUtils.end_stage(fail_message='', throw_on_fail=False)
        PrintUtils.print_extra('Operation *cancelled* by user - please wait for cleanup code to complete')
        is_user_cancelled = True
        
    except Exception as ex:
        # Optionally fail stage
        if PrintUtils.is_in_stage():
            PrintUtils.end_stage(fail_message=ex, throw_on_fail=False)
            
        # Save error and print it as an extra
        PrintUtils.print_extra(f'Error: {ex}')
        last_error = ex
        
    finally:
        # Print final status
        if last_error is not None:
            PrintUtils.print_error(f'{last_error}\n')
        elif is_user_cancelled:
            PrintUtils.print_extra(f'Operation *cancelled* by user\n')
        else:
            PrintUtils.print_extra(f'Finished successfully\n')

if __name__ == '__main__':
    main()
