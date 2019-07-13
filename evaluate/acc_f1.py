import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report

def evaluate(y_pred, y_true):
    y_pred = y_pred.detach().numpy()
    y_pred = y_pred.argmax(axis=1)
    y_true = y_true.numpy()

    f1 = f1_score(y_true, y_pred, average="macro")
    correct = np.sum((y_true == y_pred).astype(int))
    acc = correct / y_pred.shape[0]
    #print(classification_report(y_true, y_pred))
    return (acc, f1)


def qa_evaluate(y_pred, y_true):
    if len(y_pred)==3:
        start_logits, end_logits, type_logits = y_pred
        start_positions, end_positions, type_positions = y_true
        start_acc, start_f1 = evaluate(start_logits, start_positions)
        end_acc, end_f1 = evaluate(end_logits, end_positions)
        type_acc, type_f1 = evaluate(type_logits, type_positions)
        print(
            '\ntype_f1: %4f - start_f1: %4f - end_f1: %4f - type_acc:%4f - start_acc:%4f - end_acc:%4f\n'
            % (type_f1, start_f1, end_f1, type_acc, start_acc, end_acc))
    else:
        start_logits, end_logits, = y_pred
        start_positions, end_positions, = y_true
        start_acc, start_f1 = evaluate(start_logits, start_positions)
        end_acc, end_f1 = evaluate(end_logits, end_positions)

    acc = (start_acc + end_acc) / 2
    f1 = (start_f1 + end_f1) / 2
    return acc, f1