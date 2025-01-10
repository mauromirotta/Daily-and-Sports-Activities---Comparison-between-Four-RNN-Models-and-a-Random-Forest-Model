import torch.nn as nn

class StackedRNN(nn.Module):
    ''' Stacked RNN model class '''
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(StackedRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out
