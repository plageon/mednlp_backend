from transformers import EvalPrediction
import datasets
import numpy as np

def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions,axis=1)
    labels = p.label_ids
    correct = 0
    for pred, label in zip(preds, labels):
        if pred == label:
            correct += 1
    return {
        'acc': correct / len(labels)
    }
