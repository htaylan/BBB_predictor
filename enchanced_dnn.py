import torch.nn as nn

class EnhancedDNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout_rate):
        super(EnhancedDNN, self).__init__()
        layers = []
        for i in range(len(hidden_layers)):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_layers[i]))
            else:
                layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_layers[i]))
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
