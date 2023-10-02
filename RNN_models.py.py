import torch
import torch.nn as nn
import torch.nn.functional as F



class LightweightOriginal(nn.Module):
    
    def __init__(self, model_name, input_dim, hidden_dim, num_layers, lstm_dropout, final_dropout):
        super(LightweightOriginal, self).__init__()

        # Attributes:
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Layers:
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=lstm_dropout, batch_first=True) 
        self.dropout = nn.Dropout(final_dropout)
        self.decoder = nn.Linear(hidden_dim, 1) 

    def forward(self, x, h0, c0):
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.decoder(self.dropout(out[:, -1, :]))
        return out, (hn, cn)

    def init_hidden_and_cell_states(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))



#REF: https://www.kaggle.com/code/talissamoura/energy-consumption-forecast-with-lstms#3---Creating-the-model
class BidirectionalLightweightOriginal(nn.Module):
    
    def __init__(self, model_name, input_dim, hidden_dim, num_layers, lstm_dropout, final_dropout, is_bidirectional=False):
        super(BidirectionalLightweightOriginal, self).__init__()

        # Attributes:
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.is_bidirectional = is_bidirectional
        
        self.multiplier = 2 if self.is_bidirectional else 1

        # Layers:
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=lstm_dropout, batch_first=True, bidirectional=is_bidirectional) 
        self.dropout = nn.Dropout(final_dropout)
        
        # Larger size if bidirectional:
        self.decoder = nn.Linear(hidden_dim * self.multiplier, 1)

    def forward(self, x, h0=None, c0=None):
        # If no initial states provided, initialise to zeros:
        if h0 is None or c0 is None:
            h0, c0 = self.init_hidden_and_cell_states(x.size(0))
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        if self.is_bidirectional:
            out = torch.cat((out[:, -1, :self.hidden_dim], out[:, 0, self.hidden_dim:]), 1)
        else:
            out = out[:, -1, :]
        
        out = self.decoder(self.dropout(out))
        return out, (hn, cn)

    def init_hidden_and_cell_states(self, batch_size):
        # Modify initialisation based on whether bidirectional or not:
        return (torch.zeros(self.num_layers * self.multiplier, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers * self.multiplier, batch_size, self.hidden_dim))


class MediumweightLSTM(nn.Module):
    
    def __init__(self, model_name, input_dim, hidden_dim, num_layers, lstm_dropout, final_dropout, is_bidirectional=False):
        super(MediumweightLSTM, self).__init__()

        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.is_bidirectional = is_bidirectional

        self.multiplier = 2 if self.is_bidirectional else 1

        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=lstm_dropout, batch_first=True, bidirectional=is_bidirectional)
        self.lstm2 = nn.LSTM(hidden_dim * self.multiplier, hidden_dim, num_layers, dropout=lstm_dropout, batch_first=True, bidirectional=is_bidirectional)
        self.lstm3 = nn.LSTM(hidden_dim * self.multiplier, hidden_dim, num_layers, dropout=lstm_dropout, batch_first=True, bidirectional=is_bidirectional)
        self.dropout = nn.Dropout(final_dropout)
        self.decoder = nn.Linear(hidden_dim * self.multiplier, 1)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0, c0 = self.init_hidden_and_cell_states(x.size(0))

        x2, (hn, cn) = self.lstm1(x, (h0, c0))
        x2 = self.dropout(x2)
        x3, (hn, cn) = self.lstm2(x2)
        x3 = self.dropout(x3)
        x4, (hn, cn) = self.lstm3(x3)
        
        if self.is_bidirectional:
            x4 = torch.cat((x4[:, -1, :self.hidden_dim], x4[:, 0, self.hidden_dim:]), 1)
        else:
            x4 = x4[:, -1, :]
        
        output = self.decoder(self.dropout(x4))

        return output, (hn, cn)

    def init_hidden_and_cell_states(self, batch_size):
        return (torch.zeros(self.num_layers * self.multiplier, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers * self.multiplier, batch_size, self.hidden_dim))
    

    
# Not used, needs work: 
class DeepLSTM(nn.Module):
    
    def __init__(self, model_name, input_dim, hd1, hd2, hd3, num_layers, lstm_dropout, final_dropout):
        super(DeepMountainLSTM, self).__init__()

        self.model_name = model_name
        self.hd1 = hd1
        self.hd2 = hd2
        self.hd3 = hd3
        self.num_layers = num_layers

        self.lstm1 = nn.LSTM(input_dim, hd1, num_layers, dropout=lstm_dropout, batch_first=True)
        self.lstm2 = nn.LSTM(hd1, hd2, num_layers, dropout=lstm_dropout, batch_first=True)
        self.lstm3 = nn.LSTM(hd2, hd3, num_layers, dropout=lstm_dropout, batch_first=True)
        self.lstm4 = nn.LSTM(hd3, hd2, num_layers, dropout=lstm_dropout, batch_first=True)
        self.lstm5 = nn.LSTM(hd2, hd1, num_layers, dropout=lstm_dropout, batch_first=True)
        self.dropout = nn.Dropout(final_dropout)
        self.decoder = nn.Linear(hd1, 1)

    def forward(self, x):
        h0_1, c0_1 = self.init_hidden_and_cell_states(x.size(0))
        x2, (hn1, cn1) = self.lstm1(x, (h0_1, c0_1))
        x2 = self.dropout(x2)

        h0_2, c0_2 = self.init_hidden_and_cell_states(x.size(0))
        x3, (hn2, cn2) = self.lstm2(x2, (h0_2, c0_2))
        x3 = self.dropout(x3)

        h0_3, c0_3 = self.init_hidden_and_cell_states(x.size(0))
        x4, (hn3, cn3) = self.lstm3(x3, (h0_3, c0_3))
        x4 = self.dropout(x4)

        x5, (hn4, cn4) = self.lstm4(x4, (hn3, cn3))
        x5 = self.dropout(x5)

        x6, (hn5, cn5) = self.lstm5(x5, (hn4, cn4))
        x6 = x6[:, -1, :]
        output = self.decoder(self.dropout(x6))

        return output, (hn5, cn5)


    def init_hidden_and_cell_states(self, batch_size):
        h0_1, c0_1 = torch.zeros(self.num_layers, batch_size, self.hd1), torch.zeros(self.num_layers, batch_size, self.hd1)
        h0_2, c0_2 = torch.zeros(self.num_layers, batch_size, self.hd2), torch.zeros(self.num_layers, batch_size, self.hd2)
        h0_3, c0_3 = torch.zeros(self.num_layers, batch_size, self.hd3), torch.zeros(self.num_layers, batch_size, self.hd3)

        h0 = torch.cat([h0_1, h0_2, h0_3, h0_2, h0_1], dim=2) 
        c0 = torch.cat([c0_1, c0_2, c0_3, c0_2, c0_1], dim=2) 

        return h0, c0





