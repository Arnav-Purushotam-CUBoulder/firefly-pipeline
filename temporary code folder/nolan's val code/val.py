import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# Helper functions

def print_confusion_matrix(df):
    type_counts = df['type'].value_counts()
        
    tp = type_counts.get('TP', 0)
    fp = type_counts.get('FP', 0)
    fn = type_counts.get('FN', 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
    print("Detection Performance Summary")
    print("=============================")
    print(f"{'True Positives (TP):':<25} {tp}")
    print(f"{'False Positives (FP):':<25} {fp}")
    print(f"{'False Negatives (FN):':<25} {fn}")
    print("-----------------------------")
    print(f"{'Total Ground Truth Points:':<25} {tp + fn}")
    print(f"{'Total Model Detections:':<25} {tp + fp}")
    print("=============================")
    print(f"{'Precision:':<25} {precision:.4f}")
    print(f"{'Recall:':<25} {recall:.4f}")
    print(f"{'F1-Score:':<25} {f1_score:.4f}")

def classify_points(true_df, exp_df, threshold):
    result_df = exp_df.copy()
    all_tp_indices = []
    all_fn_rows = []
    common_timestamps = np.unique(np.concatenate((true_df['t'].unique(), exp_df['t'].unique())))

    for t_val in common_timestamps:
        true_points_at_t = true_df[true_df['t'] == t_val]
        exp_points_at_t = result_df[result_df['t'] == t_val]

        if true_points_at_t.empty and exp_points_at_t.empty:
            continue
                
        if true_points_at_t.empty:
            continue
                    
        if exp_points_at_t.empty:
            all_fn_rows.append(true_points_at_t)
            continue

        true_np = true_points_at_t[['x', 'y']].to_numpy()
        exp_np = exp_points_at_t[['x', 'y']].to_numpy()

        # Solve linear assignment
        C = cdist(true_np, exp_np)
        C[C > threshold] = 1e9

        true_indices, model_indices = linear_sum_assignment(C)
        valid_match_mask = C[true_indices, model_indices] < 1e9
            
        matched_true_indices = true_indices[valid_match_mask]
        matched_model_indices = model_indices[valid_match_mask]
                
        tp_global_indices = exp_points_at_t.index[matched_model_indices]
        all_tp_indices.extend(tp_global_indices)
                
        all_true_indices_at_t = np.arange(true_points_at_t.shape[0])
        fn_local_indices = np.setdiff1d(all_true_indices_at_t, matched_true_indices)
        if len(fn_local_indices) > 0:
            fn_rows = true_points_at_t.iloc[fn_local_indices]
            all_fn_rows.append(fn_rows)

    result_df['type'] = 'FP'
    if all_tp_indices:
        result_df.loc[all_tp_indices, 'type'] = 'TP'
                
    if all_fn_rows:
        fn_df = pd.concat(all_fn_rows, ignore_index=True)
        fn_df['type'] = 'FN'
        result_df = pd.concat([result_df, fn_df], ignore_index=True)
                
    return result_df

def plot_pr_roc_curves(df):
    detections = df[df['type'].isin(['TP', 'FP'])].copy()
            
    y_true = (detections['type'] == 'TP').astype(int)
    y_score = detections['firefly_logit']

    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_score)
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_score)

    avg_precision = average_precision_score(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.plot(recall, precision, color='b', label=f'AP = {avg_precision:.2f}')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision-Recall Curve')
    ax1.legend(loc='lower left')
    ax1.grid(True)

    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate (Recall)')
    ax2.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax2.legend(loc="lower right")
    ax2.grid(True)
        
    plt.tight_layout()
    plt.show()

def plot_xt_by_time(df):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(15, 8))
    
    scatter = ax.scatter(
        df['x'], 
        df['y'], 
        c=df['t'], 
        cmap='viridis', 
        s=15, 
        alpha=0.8
    )
    
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Time (t)', rotation=270, labelpad=15)
    
    ax.set_xlabel('X-coordinates')
    ax.set_ylabel('Y-coordinates')
    ax.set_title('Detected centroids over time')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()




true_df = pd.read_csv('/Users/arnavps/Desktop/New DL project data to transfer to external disk/fixing stage9 val/forresti/csv files/gt_norm_offsetplus4000.csv')
model_df_raw = pd.read_csv('/Users/arnavps/Desktop/New DL project data to transfer to external disk/fixing stage9 val/forresti/csv files/4k_to_4_5k_chopped_Forresti_C0107_fireflies_logits.csv')
DISTANCE_THRESHOLD = 10.0

resnet_results_df = classify_points(true_df, model_df_raw, DISTANCE_THRESHOLD)

print(f"FRONTALIS RESULTS: \n")
print_confusion_matrix(resnet_results_df)
plot_pr_roc_curves(resnet_results_df)