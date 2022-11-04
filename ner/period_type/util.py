from transformers import EvalPrediction
import datasets
import numpy as np


def compute_metrics(p: EvalPrediction):
    labels, preds, acc = {}, {}, {}
    labels['period0'], labels['period1'], labels['cancer_type'] = p.label_ids
    preds['period0'], preds['period1'], preds['cancer_type'] = p.predictions
    acc['acc'] = 0
    for field in ['period0', 'period1', 'cancer_type']:
        preds[field] = np.argmax(preds[field], axis=1)
        correct = 0
        for pred, label in zip(preds[field], labels[field]):
            if pred == label:
                correct += 1
        acc[field] = correct / len(labels[field])
        acc['acc'] += acc[field]
    acc['acc'] /= 3

    return acc
