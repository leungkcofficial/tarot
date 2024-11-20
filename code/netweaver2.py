import torch
import torch.nn as nn

# ANN Network Initialization
def ann_net_init(input_size, num_nodes, batch_norm=False, dropout=0.0, output_size=1):
    """
    Initializes an Artificial Neural Network (ANN).

    Args:
        input_size (int): The size of the input features.
        num_nodes (list): List specifying the number of nodes in each hidden layer.
        batch_norm (bool, optional): Whether to include batch normalization. Defaults to False.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        output_size (int, optional): Size of the output layer. Defaults to 1.

    Returns:
        nn.Sequential: The initialized ANN model.
    """
    layers = []
    for i, nodes in enumerate(num_nodes):
        layers.append(nn.Linear(input_size if i == 0 else num_nodes[i - 1], nodes))
        layers.append(nn.ReLU())
        if batch_norm:
            layers.append(nn.BatchNorm1d(nodes))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(num_nodes[-1], output_size))
    return nn.Sequential(*layers)

# Generalized ANN/DeepHit Network Initialization
def generalized_ann_net_init(input_size, num_nodes, batch_norm=False, dropout=0.0, output_size=1, num_risks=None, num_time_bins=None):
    """
    Initializes an Artificial Neural Network (ANN) for both DeepSurv and DeepHit.

    Args:
        input_size (int): The size of the input features.
        num_nodes (list): List specifying the number of nodes in each hidden layer.
        batch_norm (bool, optional): Whether to include batch normalization. Defaults to False.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        output_size (int, optional): Size of the output layer. Defaults to 1.
        num_risks (int, optional): Number of risks to predict (for DeepHit). Defaults to None.
        num_time_bins (int, optional): Number of time bins to predict (for DeepHit). Defaults to None.

    Returns:
        nn.Sequential or DHANNWrapper: The initialized ANN model or wrapped model for DeepHit.
    """
    final_output_size = output_size
    if num_risks is not None and num_time_bins is not None:
        final_output_size = num_risks * num_time_bins

    net = ann_net_init(input_size, num_nodes, batch_norm, dropout, output_size=final_output_size)
    
    if num_risks is not None and num_time_bins is not None:
        return DHANNWrapper(net, num_risks, num_time_bins)
    return net

# LSTM Network Initialization
class CustomLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_layers, dropout=0.0, batch_norm=False, output_size=1):
        """
        Initializes a custom LSTM model.

        Args:
            input_size (int): Size of the input features.
            hidden_sizes (list): List specifying the hidden sizes of each LSTM layer.
            num_layers (int): Number of LSTM layers.
            dropout (float, optional): Dropout rate between LSTM layers. Defaults to 0.0.
            batch_norm (bool, optional): Whether to include batch normalization after LSTM. Defaults to False.
            output_size (int, optional): Size of the output layer. Defaults to 1.
        """
        super(CustomLSTMNet, self).__init__()
        assert len(hidden_sizes) == num_layers, "hidden_sizes should have a length equal to num_layers"

        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_sizes[i - 1]
            out_size = hidden_sizes[i]
            self.lstm_layers.append(nn.LSTM(input_size=in_size, hidden_size=out_size, num_layers=1, batch_first=True))
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers - 1)])
        self.fc = nn.Linear(hidden_sizes[-1], output_size)
        self.batch_norm = nn.BatchNorm1d(hidden_sizes[-1]) if batch_norm else None

    def forward(self, x):
        for i, lstm in enumerate(self.lstm_layers):
            x, _ = lstm(x)
            if i < len(self.dropouts):
                x = self.dropouts[i](x)
        h_lstm_last = x[:, -1, :]
        if self.batch_norm:
            h_lstm_last = self.batch_norm(h_lstm_last)
        return self.fc(h_lstm_last)

# Wrapper for DeepHit ANN
class DHANNWrapper(nn.Module):
    def __init__(self, net, num_risks, num_time_bins):
        """
        Wrapper for ANN-based DeepHit model.

        Args:
            net (nn.Sequential): The ANN model.
            num_risks (int): Number of risks to predict.
            num_time_bins (int): Number of time bins to predict.
        """
        super(DHANNWrapper, self).__init__()
        self.net = net
        self.num_risks = num_risks
        self.num_time_bins = num_time_bins

    def forward(self, input):
        out = self.net(input)
        return out.view(out.size(0), self.num_risks, self.num_time_bins)

# Wrapper for LSTM-based DeepHit model
class LSTMWrapper(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_layers, dropout, batch_norm, num_risks, num_time_bins):
        """
        Wrapper for LSTM-based DeepHit model.

        Args:
            input_size (int): Size of the input features.
            hidden_sizes (list): List specifying the hidden sizes of each LSTM layer.
            num_layers (int): Number of LSTM layers.
            dropout (float, optional): Dropout rate between LSTM layers. Defaults to 0.0.
            batch_norm (bool, optional): Whether to include batch normalization after LSTM. Defaults to False.
            num_risks (int): Number of risks to predict.
            num_time_bins (int): Number of time bins to predict.
        """
        super(LSTMWrapper, self).__init__()
        assert len(hidden_sizes) == num_layers, "hidden_sizes should have a length equal to num_layers"

        self.num_risks = num_risks
        self.num_time_bins = num_time_bins
        self.lstm = CustomLSTMNet(input_size, hidden_sizes, num_layers, dropout, batch_norm, output_size=num_risks * num_time_bins)

    def forward(self, x):
        out = self.lstm(x)
        return out.view(out.size(0), self.num_risks, self.num_time_bins)

# Function to initialize LSTM Network for DeepHit
def lstm_net_init(input_size, num_nodes, batch_norm=False, dropout=0.0, num_risks=None, num_time_bins=None):
    """
    Initializes an LSTM model for either DeepHit or DeepSurv.

    Args:
        input_size (int): Size of the input features.
        num_nodes (list): List specifying the hidden sizes of each LSTM layer.
        batch_norm (bool, optional): Whether to include batch normalization. Defaults to False.
        dropout (float, optional): Dropout rate between LSTM layers. Defaults to 0.0.
        num_risks (int, optional): Number of risks to predict (for DeepHit). Defaults to None.
        num_time_bins (int, optional): Number of time bins to predict (for DeepHit). Defaults to None.

    Returns:
        CustomLSTMNet or LSTMWrapper: The initialized LSTM model for DeepSurv or wrapped model for DeepHit.
    """
    if num_risks is not None and num_time_bins is not None:
        # DeepHit LSTM
        return LSTMWrapper(input_size, num_nodes, len(num_nodes), dropout, batch_norm, num_risks, num_time_bins)
    else:
        # DeepSurv LSTM
        return CustomLSTMNet(input_size, num_nodes, len(num_nodes), dropout, batch_norm, output_size=1)
