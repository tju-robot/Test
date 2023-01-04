import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.u = nn.Linear(input_size, hidden_size)
        self.w = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, output_size)

        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden):

        u_x = self.u(inputs)

        hidden = self.w(hidden)
        hidden = self.tanh(hidden + u_x)

        output = self.softmax(self.v(hidden))

        return output, hidden


class LSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, out_features):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.output_size = out_features
        # i_t
        self.U_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_i = nn.Parameter(torch.Tensor(hidden_sz))

        # f_t
        self.U_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_f = nn.Parameter(torch.Tensor(hidden_sz))

        # c_t
        self.U_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.Tensor(hidden_sz))

        # o_t
        self.U_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        self.V_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.Tensor(hidden_sz))

        self.Linear = nn.Linear(hidden_sz, out_features)  # 全连接层做输出
        self.reset()

    def reset(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, init_states=None):

        bs, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (
                torch.zeros(bs, self.hidden_size).to(x.device),
                torch.zeros(bs, self.hidden_size).to(x.device),
            )
        else:
            h_t, c_t = init_states

        for t in range(seq_sz):
            x_t = x[:, t, :]

            i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.V_i + self.b_i)
            f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.V_f + self.b_f)
            g_t = torch.tanh(x_t @ self.U_c + h_t @ self.V_c + self.b_c)
            o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.V_o + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            y = self.Linear(h_t)
            hidden_seq.append(y)

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)


class GRU(nn.Module):
    def __init__(self, in_features, hid_features, out_features):
        super(GRU, self).__init__()
        self.indim = in_features
        self.hidim = hid_features
        self.outdim = out_features
        self.W_zh, self.W_zx, self.b_z = self.get_three_parameters()
        self.W_rh, self.W_rx, self.b_r = self.get_three_parameters()
        self.W_hh, self.W_hx, self.b_h = self.get_three_parameters()
        self.Linear = nn.Linear(hid_features, out_features)  # 全连接层做输出
        self.reset()

    def forward(self, input, state):
        input = input.type(torch.float32)
        if torch.cuda.is_available():
            input = input.cuda()
        Y = []
        h = state
        h = h.cuda()
        for x in input:
            z = F.sigmoid(h @ self.W_zh + x @ self.W_zx + self.b_z)
            r = F.sigmoid(h @ self.W_rh + x @ self.W_rx + self.b_r)
            ht = F.tanh((h * r) @ self.W_hh + x @ self.W_hx + self.b_h)
            h = (1 - z) * h + z * ht
            y = self.Linear(h)
            Y.append(y)
        return torch.cat(Y, dim=0), h

    def get_three_parameters(self):
        indim, hidim, outdim = self.indim, self.hidim, self.outdim
        return nn.Parameter(torch.FloatTensor(hidim, hidim)), \
               nn.Parameter(torch.FloatTensor(indim, hidim)), \
               nn.Parameter(torch.FloatTensor(hidim))

    def reset(self):
        stdv = 1.0 / math.sqrt(self.hidim)
        for param in self.parameters():
            nn.init.uniform_(param, -stdv, stdv)