import numpy as np
from sklearn.model_selection import train_test_split, KFold, ParameterGrid
import torch
from utils.data_preprocessing import load_and_preprocess_data
from models.classifier import PyTorchDNNClassifier
import pickle
import joblib


def main():
    X, y = load_and_preprocess_data('data/descriptors.npy', 'data/labels.npy')
    
    joblib.dump(pca, 'pca_model.pkl')
    joblib.dump(scaler, 'scaler_model.pkl')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    input_size = X.shape[1]
    output_size = 2
    param_grid = {
        'hidden_layers': [
            [100, 200, 300, 500, 800],
            [100, 150, 200, 250, 300],
            [50, 100, 150, 200, 250, 300]
        ],
        'dropout_rate': [0.2, 0.3, 0.4],
        'learning_rate': [0.0001, 0.001, 0.01]
    }
    epochs = 500
    patience = 20
    best_score = 0
    best_model = None
    best_params = None

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for params in ParameterGrid(param_grid):
        scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        roc_auc_scores = []

        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
    
            model = PyTorchDNNClassifier(input_size, params['hidden_layers'], output_size, params['dropout_rate'], params['learning_rate'])
            model.fit(X_train, y_train, X_val, y_val, epochs=epochs, patience=patience)
            y_pred = model.predict(X_val)
    
            scores.append(accuracy_score(y_val, y_pred))
            precision_scores.append(precision_score(y_val, y_pred))
            recall_scores.append(recall_score(y_val, y_pred))
            f1_scores.append(f1_score(y_val, y_pred))
            roc_auc_scores.append(roc_auc_score(y_val, y_pred))
    
        avg_score = np.mean(scores)
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_f1 = np.mean(f1_scores)
        avg_roc_auc = np.mean(roc_auc_scores)
    
        if avg_score > best_score:
            best_score = avg_score
            best_model = model
            best_params = params

    torch.save(best_model.model.state_dict(), 'best_model_dnn.pth')
    
    print("Best score:", best_score)
    print("Best parameters:", best_params)
    
    # Evaluate the final model
    final_accuracy, final_precision, final_recall, final_f1, final_roc_auc, false_positives, false_negatives, true_positives, true_negatives = best_model.evaluate(X_test, y_test)
    
    print("Final accuracy:", final_accuracy)
    print("Final precision:", final_precision)
    print("Final recall:", final_recall)
    print("Final F1 score:", final_f1)
    print("Final ROC AUC score:", final_roc_auc)
    
    # Save metrics and best parameters to a file
    best_model.save_metrics('best_model_metrics3.dat',
                            {'accuracy': final_accuracy, 'precision': final_precision, 'recall': final_recall, 'f1': final_f1, 'roc_auc': final_roc_auc},
                            best_params, false_positives, false_negatives, true_positives, true_negatives)


if __name__ == "__main__":
  main()
