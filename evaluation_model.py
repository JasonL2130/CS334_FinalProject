from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, average_precision_score, f1_score

def eval_model(estimator, X_test, y_test, yHat, elapsed_time):
    # Function that returns evaluation metrics for a machine learning model.
    yHat_prob = estimator.predict_proba(X_test)[:, 1]
    accuracy = get_accuracy(yHat, y_test)
    auc = roc_auc_score(y_test, yHat_prob)
    auprc = average_precision_score(y_test, yHat_prob)
    f1 = f1_score(y_test, yHat)
    fpr, tpr, thresholds = roc_curve(y_test, yHat_prob)
    time = elapsed_time

    resultDict = { 'AUC': auc, 'AUPRC': auprc, 'F1': f1, 'Accuracy': accuracy, 'Time': time}
    roc = {'fpr': fpr, 'tpr': tpr}

    return resultDict, roc

def get_accuracy(yHat, yTest):
    # Returns the accuracy of a model.
    return accuracy_score(yHat, yTest)
