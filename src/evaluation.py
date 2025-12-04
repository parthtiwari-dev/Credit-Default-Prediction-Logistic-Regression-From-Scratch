# src/evaluation.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import Tuple, List
import os

def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int,int,int,int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn, fp, fn, tp

def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float,float,float]:
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return precision, recall, f1, accuracy

def roc_curve_from_probs(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # returns fpr, tpr, thresholds
    desc_score_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[desc_score_indices]
    y_prob_sorted = y_prob[desc_score_indices]
    thresholds = np.concatenate(([y_prob_sorted[0] + 1e-8], y_prob_sorted, [y_prob_sorted[-1] - 1e-8]))
    tprs = []
    fprs = []
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        tpr = tp / P if P > 0 else 0.0
        fpr = fp / N if N > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)
    return np.array(fprs), np.array(tprs), thresholds

def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    # assume x sorted ascending
    return np.trapz(y, x)

def precision_recall_curve_from_probs(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # returns precision, recall, thresholds
    desc_score_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[desc_score_indices]
    y_prob_sorted = y_prob[desc_score_indices]
    thresholds = np.unique(y_prob_sorted)
    precisions = []
    recalls = []
    P = np.sum(y_true == 1)
    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / P if P > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
    return np.array(precisions), np.array(recalls), thresholds

def plot_roc(fpr, tpr, auc_score=None, save_path=None):
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1],[0,1],'--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    title = 'ROC Curve' + (f' (AUC={auc_score:.4f})' if auc_score is not None else '')
    plt.title(title)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_pr(precision, recall, ap=None, save_path=None):
    plt.figure()
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    title = 'Precision-Recall Curve' + (f' (AP={ap:.4f})' if ap is not None else '')
    plt.title(title)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def threshold_vs_metrics(y_true, y_prob, thresholds, save_path=None):
    precisions = []
    recalls = []
    f1s = []
    accuracies = []
    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        p, r, f1, acc = precision_recall_f1(y_true, preds)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        accuracies.append(acc)
    plt.figure()
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1s, label='F1')
    plt.plot(thresholds, accuracies, label='Accuracy')
    plt.xlabel('Threshold')
    plt.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
