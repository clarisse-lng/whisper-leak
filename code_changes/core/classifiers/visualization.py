import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, 
                            confusion_matrix, classification_report, average_precision_score,
                            RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from core.utils import PrintUtils

def set_plot_style():
    """Set plot style for attractive visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("deep")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, best_epoch, output_file='training_curves.png'):
    """Plot training and validation loss/accuracy curves."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
    plt.axvline(x=best_epoch, color='g', linestyle='--', label='Best Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accs) + 1), train_accs, 'b-', label='Training Accuracy')
    plt.plot(range(1, len(val_accs) + 1), val_accs, 'r-', label='Validation Accuracy')
    plt.axvline(x=best_epoch, color='g', linestyle='--', label='Best Model')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    PrintUtils.print_extra(f"Training curves saved to {output_file}")

    # Write data to CSV for further analysis
    training_data = {
        'Epoch': range(1, len(train_losses) + 1),
        'Train Loss': train_losses,
        'Validation Loss': val_losses,
        'Train Accuracy': train_accs,
        'Validation Accuracy': val_accs,
    }
    training_df = pd.DataFrame(training_data)
    output_curve = os.path.splitext(output_file)[0] + '.csv'
    training_df.to_csv(output_curve, index=False)


def plot_roc_curve(y_true, y_scores, output_file='roc_curve.png'):
    """Plot ROC curve. Handles binary (1D scores) and multiclass (2D scores, OvR)."""
    from sklearn.preprocessing import label_binarize

    y_true = np.array(y_true)
    classes = sorted(np.unique(y_true))
    n_classes = len(classes)

    plt.figure(figsize=(10, 8))

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    else:
        y_bin = label_binarize(y_true, classes=classes)
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        all_fpr = np.unique(np.concatenate([
            roc_curve(y_bin[:, i], y_scores[:, i])[0] for i in range(n_classes)
        ]))
        mean_tpr = np.zeros_like(all_fpr)
        for i, cls in enumerate(classes):
            fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_scores[:, i])
            mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
            auc_i = auc(fpr_i, tpr_i)
            plt.plot(fpr_i, tpr_i, lw=1, alpha=0.5, color=colors[i], label=f'Class {cls} (AUC={auc_i:.2f})')
        mean_tpr /= n_classes
        roc_auc = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, color='black', lw=2, linestyle='--',
                 label=f'Macro avg (AUC={roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    PrintUtils.print_extra(f"ROC curve saved to {output_file}")

    return roc_auc


def plot_precision_recall_curve(y_true, y_scores, output_file='precision_recall_curve.png'):
    """Plot Precision-Recall curve. Handles binary (1D) and multiclass (2D, OvR)."""
    from sklearn.preprocessing import label_binarize

    y_true = np.array(y_true)
    classes = sorted(np.unique(y_true))
    n_classes = len(classes)

    plt.figure(figsize=(10, 8))

    if n_classes == 2:
        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        plt.plot(rec, prec, color='green', lw=2,
                 label=f'PR curve (AP = {avg_precision:.3f})')
        plt.axhline(y=y_true.mean(), color='navy', linestyle='--', label='Random')
    else:
        y_bin = label_binarize(y_true, classes=classes)
        colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        ap_scores = []
        for i, cls in enumerate(classes):
            prec_i, rec_i, _ = precision_recall_curve(y_bin[:, i], y_scores[:, i])
            ap_i = average_precision_score(y_bin[:, i], y_scores[:, i])
            ap_scores.append(ap_i)
            plt.plot(rec_i, prec_i, lw=1, alpha=0.5, color=colors[i],
                     label=f'Class {cls} (AP={ap_i:.2f})')
        avg_precision = float(np.mean(ap_scores))
        plt.axhline(y=avg_precision, color='black', lw=2, linestyle='--',
                    label=f'Macro avg AP={avg_precision:.3f}')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve', fontweight='bold')
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    PrintUtils.print_extra(f"Precision-Recall curve saved to {output_file}")

    return avg_precision


def plot_confusion_matrix(y_true, y_pred, output_file='confusion_matrix.png', class_names=None):
    """Plot confusion matrix (works for binary and multiclass)."""
    conf_matrix = confusion_matrix(y_true, y_pred)
    n_cls = conf_matrix.shape[0]

    if class_names is None:
        class_names = ['Negative', 'Positive'] if n_cls == 2 else [str(i) for i in range(n_cls)]

    fig_size = max(8, n_cls * 1.5)
    plt.figure(figsize=(fig_size, fig_size * 0.8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names,
                yticklabels=class_names)

    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    PrintUtils.print_extra(f"Confusion matrix saved to {output_file}")

    return conf_matrix


def plot_score_distribution(scores, labels=None, output_file='prediction_score_distribution.png'):
    """
    Plot distribution of prediction scores with different colors for positive and negative targets.
    
    Parameters:
    scores (array-like): Model prediction scores
    labels (array-like, optional): True labels corresponding to scores. If provided, scores will be colored by class.
    output_file (str): Path to save the output image
    """
    plt.figure(figsize=(10, 6))
    
    if labels is not None:
        # Convert to numpy arrays if they aren't already
        scores = np.array(scores)
        labels = np.array(labels)
        
        # Create a DataFrame for seaborn to use for stacked histograms
        import pandas as pd
        df = pd.DataFrame({
            'score': scores,
            'class': ['Positive Class' if label == 1 else 'Negative Class' for label in labels]
        })
        
        # Plot stacked distributions
        sns.histplot(data=df, x='score', hue='class', bins=50, 
                    multiple="stack", palette={'Negative Class': 'royalblue', 'Positive Class': 'crimson'})
    else:
        # If no labels are provided, just plot all scores in one color
        sns.histplot(scores, bins=50, color='royalblue', label='All Predictions')
    
    plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold (0.5)')
    plt.xlabel('Prediction Score')
    plt.ylabel('Count')
    plt.title('Distribution of Model Prediction Scores', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    PrintUtils.print_extra(f"Prediction score distribution saved to {output_file}")


def create_model_dashboard(val_scores, val_labels, train_losses, val_losses, best_epoch,
                          output_file='model_performance_dashboard.png'):
    """Create a dashboard of model performance metrics. Handles binary and multiclass."""
    from sklearn.preprocessing import label_binarize

    val_labels = np.array(val_labels)
    classes = sorted(np.unique(val_labels))
    n_classes = len(classes)

    if n_classes > 2:
        val_preds = np.argmax(val_scores, axis=1)
    else:
        val_preds = (val_scores > 0.5).astype(int)

    conf_matrix = confusion_matrix(val_labels, val_preds)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    plt.figure(figsize=(16, 12))

    # ROC Curve (top left)
    plt.subplot(2, 2, 1)
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(val_labels, val_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    else:
        y_bin = label_binarize(val_labels, classes=classes)
        all_fpr = np.unique(np.concatenate([
            roc_curve(y_bin[:, i], val_scores[:, i])[0] for i in range(n_classes)
        ]))
        mean_tpr = np.zeros_like(all_fpr)
        for i, cls in enumerate(classes):
            fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], val_scores[:, i])
            mean_tpr += np.interp(all_fpr, fpr_i, tpr_i)
            plt.plot(fpr_i, tpr_i, lw=1, alpha=0.4, color=colors[i])
        mean_tpr /= n_classes
        roc_auc = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, color='black', lw=2, linestyle='--',
                 label=f'Macro AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # Precision-Recall Curve (top right)
    plt.subplot(2, 2, 2)
    if n_classes == 2:
        prec, rec, _ = precision_recall_curve(val_labels, val_scores)
        avg_precision = average_precision_score(val_labels, val_scores)
        plt.plot(rec, prec, color='green', lw=2, label=f'AP = {avg_precision:.3f}')
        plt.axhline(y=val_labels.mean(), color='navy', linestyle='--', label='Random')
    else:
        y_bin = label_binarize(val_labels, classes=classes)
        ap_scores = []
        for i, cls in enumerate(classes):
            prec_i, rec_i, _ = precision_recall_curve(y_bin[:, i], val_scores[:, i])
            ap_i = average_precision_score(y_bin[:, i], val_scores[:, i])
            ap_scores.append(ap_i)
            plt.plot(rec_i, prec_i, lw=1, alpha=0.4, color=colors[i])
        avg_precision = float(np.mean(ap_scores))
        plt.axhline(y=avg_precision, color='black', lw=2, linestyle='--',
                    label=f'Macro AP = {avg_precision:.3f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    # Confusion Matrix (bottom left)
    plt.subplot(2, 2, 3)
    class_names = ['Negative', 'Positive'] if n_classes == 2 else [str(c) for c in classes]
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')

    # Training/Validation Loss Curve (bottom right)
    plt.subplot(2, 2, 4)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation')
    if best_epoch:
        plt.axvline(x=best_epoch, color='g', linestyle='--', label='Best Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.suptitle('Model Performance Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    PrintUtils.print_extra(f"Model performance dashboard saved to {output_file}")


def calculate_metrics(test_labels, test_scores, test_preds, conf_matrix, df):
        """
        Calculate benchmark metrics for a chatbot.
        Supports both binary (test_scores 1D) and multiclass (test_scores 2D) modes.

        Args:
            test_labels: True labels
            test_scores: 1D probability scores (binary) or 2D probability matrix (multiclass)
            test_preds: Predicted class labels
            conf_matrix: Confusion matrix
            df: Original dataframe with all data

        Returns:
            dict: Dictionary of calculated metrics
        """
        from sklearn.preprocessing import label_binarize

        test_labels = np.array(test_labels)
        test_preds = np.array(test_preds)
        n_classes = len(np.unique(test_labels))

        # Calculate data statistics
        data_lengths = df['data_lengths'].apply(len)
        median_data_length = np.median(data_lengths)
        avg_data_length = np.mean(data_lengths)
        stddev_data_length = np.std(data_lengths)
        all_data_sizes = np.concatenate(df['data_lengths'].values)
        median_data_size = np.median(all_data_sizes)
        avg_data_size = np.mean(all_data_sizes)
        stddev_data_size = np.std(all_data_sizes)

        accuracy = accuracy_score(test_labels, test_preds)

        if n_classes == 2:
            # --- Binary mode ---
            tn, fp, fn, tp = conf_matrix.ravel()
            auc_score = roc_auc_score(test_labels, test_scores)
            precision_curve, recall_curve, _ = precision_recall_curve(test_labels, test_scores)
            auprc = auc(recall_curve, precision_curve)
            f1 = f1_score(test_labels, test_preds)
            recall = recall_score(test_labels, test_preds)
            precision = precision_score(test_labels, test_preds)

            precision_at_recall = {}
            for r in [0.05] + list(np.arange(0.1, 1.1, 0.1)):
                precision_at_recall[r] = np.interp(r, recall_curve[::-1], precision_curve[::-1])
            for r, p in precision_at_recall.items():
                PrintUtils.print_extra(f"Precision at {r:.2f} recall: {p:.3f}")

            metrics = {
                'AUC': auc_score,
                'AUPRC': auprc,
                'F1 Score': f1,
                'Recall': recall,
                'Precision': precision,
                'Accuracy': accuracy,
                'Total': len(test_labels),
                'Positives': int((test_labels == 1).sum()),
                'Negatives': int((test_labels == 0).sum()),
                'True Positives': int(tp),
                'True Negatives': int(tn),
                'False Positives': int(fp),
                'False Negatives': int(fn),
            }
            for r, p in precision_at_recall.items():
                metrics[f'Precision at {r:.2f} Recall'] = p

        else:
            # --- Multiclass mode ---
            classes = sorted(np.unique(test_labels))
            auc_score = roc_auc_score(test_labels, test_scores, multi_class='ovr', average='macro')
            y_bin = label_binarize(test_labels, classes=classes)
            auprc = float(np.mean([
                average_precision_score(y_bin[:, i], test_scores[:, i])
                for i in range(len(classes))
            ]))
            f1 = f1_score(test_labels, test_preds, average='macro')
            recall = recall_score(test_labels, test_preds, average='macro')
            precision = precision_score(test_labels, test_preds, average='macro')

            metrics = {
                'AUC (macro OvR)': auc_score,
                'AUPRC (macro)': auprc,
                'F1 Score (macro)': f1,
                'Recall (macro)': recall,
                'Precision (macro)': precision,
                'Accuracy': accuracy,
                'Total': len(test_labels),
                'Num Classes': n_classes,
            }
            for cls in classes:
                metrics[f'Count class {cls}'] = int((test_labels == cls).sum())

        # Shared data statistics
        metrics.update({
            'Median Number of Network Events': median_data_length,
            'Avg Number of Network Events': avg_data_length,
            'StdDev Number of Network Events': stddev_data_length,
            'Median Network Event Size': median_data_size,
            'Avg Network Event Size': avg_data_size,
            'StdDev Network Event Size': stddev_data_size,
        })

        if "response_tokens" in df.columns:
            median_tokens = np.median(df['response_tokens'].apply(len))
            avg_tokens = np.mean(df['response_tokens'].apply(len))
            stddev_tokens = np.std(df['response_tokens'].apply(len))
            all_token_strings = np.concatenate(df['response_tokens'].values)
            token_lengths = [len(token) for token in all_token_strings]
            metrics['Median Count of Response Chunks'] = median_tokens
            metrics['Avg Count of Response Chunks'] = avg_tokens
            metrics['StdDev Count of Response Chunks'] = stddev_tokens
            metrics['Mean Length of Response Chunks'] = np.mean(token_lengths)
            metrics['Median Length of Response Chunks'] = np.median(token_lengths)

        return metrics


