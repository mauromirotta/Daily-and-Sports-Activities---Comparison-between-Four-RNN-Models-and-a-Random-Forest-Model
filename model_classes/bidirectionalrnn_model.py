import torch.nn as nn

class BidirectionalRNN(nn.Module):
    ''' Bidirectional RNN model class'''
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(BidirectionalRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, bidirectional=True, batch_first=True, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size * 2, output_size) 

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :]) 
        return out