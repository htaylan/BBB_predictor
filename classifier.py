import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pickle
from .enhanced_dnn.py import EnhancedDNN

class PyTorchDNNClassifier:
    def __init__(self, input_size, hidden_layers, output_size, dropout_rate, learning_rate):
        self.model = EnhancedDNN(input_size, hidden_layers, output_size, dropout_rate).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-2)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)

    def fit(self, X, y, X_val, y_val, epochs=epochs, patience=patience):
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_loss = float('inf')
        patience_counter = 0

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            train_accuracy = 100 * correct_train / total_train
            val_loss = self.validate(val_dataloader)
            self.scheduler.step(val_loss)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}, Training Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    def validate(self, val_dataloader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
        return val_loss / len(val_dataloader)

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def evaluate(self, X_test, y_test):
        self.model.eval()
        inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
        labels = torch.tensor(y_test, dtype=torch.long).to(device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu().numpy()
        y_test = y_test

        accuracy = accuracy_score(y_test, predicted)
        precision = precision_score(y_test, predicted)
        recall = recall_score(y_test, predicted)
        f1 = f1_score(y_test, predicted)
        roc_auc = roc_auc_score(y_test, predicted)

        false_positives = np.where((predicted == 1) & (y_test == 0))[0]
        false_negatives = np.where((predicted == 0) & (y_test == 1))[0]
        true_positives = np.where((predicted == 1) & (y_test == 1))[0]
        true_negatives = np.where((predicted == 0) & (y_test == 0))[0]

        return accuracy, precision, recall, f1, roc_auc, false_positives, false_negatives, true_positives, true_negatives

    def save_metrics(self, filename, metrics, best_params, false_positives, false_negatives, true_positives, true_negatives):
        with open(filename, 'wb') as f:
            pickle.dump({'metrics': metrics, 'best_params': best_params, 'false_positives': false_positives, 'false_negatives': false_negatives,
                         'true_positives': true_positives, 'true_negatives' : true_negatives}, f)
