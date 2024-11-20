import torch

# ANN
def ann_net_init(input_size, num_nodes, batch_norm, dropout):
    layers = []
    layers.append(torch.nn.Linear(input_size, num_nodes[0]))
    layers.append(torch.nn.ReLU())
    if batch_norm:
        layers.append(torch.nn.BatchNorm1d(num_nodes[0]))
    if dropout > 0:
        layers.append(torch.nn.Dropout(dropout))
    for i in range(1, len(num_nodes)):
        layers.append(torch.nn.Linear(num_nodes[i-1], num_nodes[i]))
        layers.append(torch.nn.ReLU())
        if batch_norm:
            layers.append(torch.nn.BatchNorm1d(num_nodes[i]))
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))
    layers.append(torch.nn.Linear(num_nodes[-1], 1))
    net = torch.nn.Sequential(*layers)
    return net

# LSTM
class CustomLSTMNet(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, num_layers, dropout, batch_norm):
        super(CustomLSTMNet, self).__init__()
        assert len(hidden_sizes) == num_layers, "hidden_sizes should have a length equal to num_layers"
        
        self.lstm_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_sizes[i-1]
            out_size = hidden_sizes[i]
            lstm_layer = torch.nn.LSTM(input_size=in_size, hidden_size=out_size, num_layers=1, batch_first=True)
            self.lstm_layers.append(lstm_layer)
        
        self.dropouts = torch.nn.ModuleList([torch.nn.Dropout(dropout) for _ in range(num_layers - 1)])
        
        self.fc = torch.nn.Linear(hidden_sizes[-1], 1)
        self.batch_norm = torch.nn.BatchNorm1d(hidden_sizes[-1]) if batch_norm else None

    def forward(self, x):
        for i, lstm in enumerate(self.lstm_layers):
            x, _ = lstm(x)
            if i < len(self.dropouts):  # Apply dropout after each layer except the last one
                x = self.dropouts[i](x)
        
        h_lstm_last = x[:, -1, :]
        if self.batch_norm:
            h_lstm_last = self.batch_norm(h_lstm_last)
        out = self.fc(h_lstm_last)
        return out

def lstm_net_init(input_size, num_nodes, batch_norm, dropout):
    net = CustomLSTMNet(input_size=input_size, hidden_sizes=num_nodes, num_layers=len(num_nodes), dropout=dropout, batch_norm=batch_norm)
    return net

# DeepHit ANN
def dh_ann_net_init(input_size, num_nodes, batch_norm, dropout, num_risks, num_time_bins):
    layers = []
    layers.append(torch.nn.Linear(input_size, num_nodes[0]))
    layers.append(torch.nn.ReLU())
    if batch_norm:
        layers.append(torch.nn.BatchNorm1d(num_nodes[0]))
    if dropout > 0:
        layers.append(torch.nn.Dropout(dropout))
    
    for i in range(1, len(num_nodes)):
        layers.append(torch.nn.Linear(num_nodes[i-1], num_nodes[i]))
        layers.append(torch.nn.ReLU())
        if batch_norm:
            layers.append(torch.nn.BatchNorm1d(num_nodes[i]))
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))

    # Output layer for num_risks * num_time_bins
    layers.append(torch.nn.Linear(num_nodes[-1], num_risks * num_time_bins))

    # Wrap the layers into a Sequential model
    net = torch.nn.Sequential(*layers)
    return net

class DHANNWrapper(torch.nn.Module):
    def __init__(self, net, num_risks, num_time_bins):
        super(DHANNWrapper, self).__init__()
        self.net = net
        self.num_risks = num_risks
        self.num_time_bins = num_time_bins

    def forward(self, input):
        # Get the output from the sequential network
        out = self.net(input)
        # Reshape the output to (batch_size, num_risks, num_time_bins)
        out = out.view(out.size(0), self.num_risks, self.num_time_bins)
        return out
    
class LSTMWrapper(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, num_layers, dropout, batch_norm, num_risks, num_time_bins):
        super(LSTMWrapper, self).__init__()
        assert len(hidden_sizes) == num_layers, "hidden_sizes should have a length equal to num_layers"
        
        self.num_risks = num_risks
        self.num_time_bins = num_time_bins
        
        # Create LSTM layers
        self.lstm_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_sizes[i-1]
            out_size = hidden_sizes[i]
            lstm_layer = torch.nn.LSTM(input_size=in_size, hidden_size=out_size, num_layers=1, batch_first=True)
            self.lstm_layers.append(lstm_layer)
        
        # Optional dropout between LSTM layers
        self.dropouts = torch.nn.ModuleList([torch.nn.Dropout(dropout) for _ in range(num_layers - 1)])

        # Fully connected output layer to predict num_risks * num_time_bins
        self.fc = torch.nn.Linear(hidden_sizes[-1], num_risks * num_time_bins)

        # Optional batch normalization
        self.batch_norm = torch.nn.BatchNorm1d(hidden_sizes[-1]) if batch_norm else None

    def forward(self, x):
        # x shape: [batch_size, seq_length, input_size]
        for i, lstm in enumerate(self.lstm_layers):
            x, _ = lstm(x)
            if i < len(self.dropouts):  # Apply dropout between LSTM layers
                x = self.dropouts[i](x)
        
        # Extract the last hidden state (we care about the final output of the LSTM)
        h_lstm_last = x[:, -1, :]  # Shape: [batch_size, hidden_size]
        
        if self.batch_norm:
            h_lstm_last = self.batch_norm(h_lstm_last)

        # Fully connected layer to predict num_risks * num_time_bins
        out = self.fc(h_lstm_last)  # Shape: [batch_size, num_risks * num_time_bins]

        # Reshape to [batch_size, num_risks, num_time_bins]
        out = out.view(out.size(0), self.num_risks, self.num_time_bins)
        
        return out
