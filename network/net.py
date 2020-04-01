import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, in_dim, hidden_dim, layer_dim, out_dim):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim

        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(input_size=in_dim,
                            hidden_size=hidden_dim,
                            num_layers=layer_dim,
                            batch_first=False)
        nn.LSTMCell

        self.fc = nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, x):
        '''

        :param x: x.shape                   seq_dim,batch_size,in_dim
        :param:h0 ho.shape                  layer_dim, batch_size,hidden_dim
        :param:c0 co.shape                  layer_dim, batch_size,hidden_dim
        :return: RNN return shape           seq_dim,batch_size,hidden_dim

        So, we need index hidden state of last time step
        Q : How can I do ?
        A : out[27,:,:]  that's it.

        BTW, We can simplify the expression very handy way :

        N_1 Simplify   out[-1,:,:]   use slice

        N_2 Simplify   out[-1,...]   use ellipsis
        '''

        h0 = torch.zeros(self.layer_dim, x.size(1), self.hidden_dim)

        c0 = torch.zeros(self.layer_dim, x.size(1), self.hidden_dim)

        out, (hn,cn) = self.lstm(x, (h0, c0))

        out = out[-1, ...]

        out = self.fc(out)

        return out
