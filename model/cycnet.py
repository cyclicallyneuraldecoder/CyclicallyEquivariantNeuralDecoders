import torch
import torch.nn as nn
import numpy as np


class cycnet(nn.Module):
    def __init__(self, conf, device):
        super(cycnet, self).__init__()
        self.clip_tanh = 10
        self.v_size = conf["data"]['v_size']
        self.e_size = conf["data"]['e_size']
        self.l_size = self.e_size // self.v_size
        self.mask_e = (torch.ones(self.l_size, self.l_size) -
                       torch.eye(self.l_size)).to(device)

        self.oddw_v1 = nn.Parameter(torch.randn(1, self.l_size))
        self.oddw_e1 = nn.Parameter(torch.randn(self.l_size, self.l_size))
        self.oddw_v2 = nn.Parameter(torch.randn(1, self.l_size))
        self.oddw_e2 = nn.Parameter(torch.randn(self.l_size, self.l_size))
        self.oddw_v3 = nn.Parameter(torch.randn(1, self.l_size))
        self.oddw_e3 = nn.Parameter(torch.randn(self.l_size, self.l_size))
        self.oddw_v4 = nn.Parameter(torch.randn(1, self.l_size))
        self.oddw_e4 = nn.Parameter(torch.randn(self.l_size, self.l_size))
        self.oddw_v5 = nn.Parameter(torch.randn(1, self.l_size))
        self.oddw_e5 = nn.Parameter(torch.randn(self.l_size, self.l_size))

        self.w_e_out = nn.Parameter(torch.randn(self.l_size))
        self.train_message = torch.zeros(
            conf["para"]["train_batch_size"], self.e_size).to(device)
        self.test_message = torch.zeros(
            conf["para"]["test_batch_size"], self.e_size).to(device)
        # To generate permutations_rowtocol, permutations_coltorow
        H = np.loadtxt(conf["data"]["H_path"])

        count = 0
        pos_cyclic_col = np.zeros([self.v_size, self.v_size])
        for i in range(self.v_size):
            for j in range(self.v_size):
                if H[(j+i) % self.v_size][i] == 1:
                    pos_cyclic_col[(j+i) % self.v_size, i] = count
                    count = count + 1
        count = 0
        pos_cyclic_row = np.zeros([self.v_size, self.v_size])
        for i in range(self.v_size):
            for j in range(self.v_size):
                if H[i][(j+i) % self.v_size] == 1:
                    pos_cyclic_row[i, (j+i) % self.v_size] = count
                    count = count + 1

        cycrowtocyccol = np.zeros(self.e_size)
        for i in range(self.v_size):
            for j in range(self.v_size):
                if H[i][j] == 1:
                    cycrowtocyccol[np.int(
                        pos_cyclic_col[i][j])] = pos_cyclic_row[i][j]

        cyccoltocycrow = np.zeros(self.e_size)
        for i in range(self.v_size):
            for j in range(self.v_size):
                if H[i][j] == 1:
                    cyccoltocycrow[np.int(
                        pos_cyclic_row[i][j])] = pos_cyclic_col[i][j]

        self.permutations_cycrowtocyccol = torch.tensor(
            np.zeros([len(cycrowtocyccol), len(cycrowtocyccol)])).to(device)
        count = 0
        for j in cycrowtocyccol:
            self.permutations_cycrowtocyccol[np.int(j)][count] = 1
            count = count + 1

        self.permutations_cyccoltocycrow = torch.tensor(
            np.zeros([len(cyccoltocycrow), len(cyccoltocycrow)])).to(device)
        count = 0
        for j in cyccoltocycrow:
            self.permutations_cyccoltocycrow[np.int(j)][count] = 1
            count = count + 1

    def odd_layer(self, inputs_v, inputs_e, oddw_v, oddw_e):
        inputs_v = inputs_v.to(torch.float)
        inputs_v = inputs_v.unsqueeze(2)
        # batch_size * v_size * l_size = (batch_size * v_size * 1) * ( 1 * l )
        v_out = torch.matmul(inputs_v, oddw_v)
        # inputs_v count by column  b*e = b*v*l
        v_out = v_out.reshape(-1, self.e_size)

        # To do cycrow to cyccolumn: b * e_size = (b * e_size) * (e_size * e*size)
        inputs_e = torch.matmul(inputs_e.to(
            torch.float), self.permutations_cycrowtocyccol.to(torch.float))
        # b*e = b*v*l * l*l
        mask_w_e = torch.mul(oddw_e, self.mask_e)
        inputs_e = inputs_e.view(-1, self.v_size, self.l_size,).to(torch.float)
        e_out = torch.matmul(inputs_e, mask_w_e)
        e_out = e_out.view(-1, self.e_size)

        # add v_out and e_out
        odd = v_out + e_out
        odd = 0.5 * torch.clamp(odd, min=-self.clip_tanh, max=self.clip_tanh)
        odd = torch.tanh(odd)
        return odd

    def even_layer(self, odd, flag_clip):
        # To do column to row
        even = torch.matmul(odd.to(torch.float),
                            self.permutations_cyccoltocycrow.to(torch.float))
        # Cumulative product then divide itself
        even = even.view(-1, self.v_size, self.l_size)
        # Matrix value:0->1
        even = torch.add(even, 1 - (torch.abs(even) > 0).to(torch.float))
        prod_even = torch.prod(even, -1)
        even = torch.div(prod_even.unsqueeze(2).repeat(
            1, 1, self.l_size), even).reshape(-1, self.e_size)
        if flag_clip:
            even = torch.clamp(even, min=-self.clip_tanh, max=self.clip_tanh)
        even = torch.log(torch.div(1 + even, 1 - even))
        return even

    def output_layer(self, inputs_v, inputs_e):
        out_layer1 = torch.matmul(inputs_e.to(
            torch.float), self.permutations_cycrowtocyccol.to(torch.float))
        out_layer2 = out_layer1.to(torch.float)
        # b*v = (b*e) * (e*v)
        out_layer3 = out_layer2.view(-1, self.v_size, self.l_size)
        # b*v = (b*v*l) * (l)
        e_out = torch.matmul(out_layer3, self.w_e_out)
        v_out = inputs_v.to(torch.float)
        return v_out + e_out

    def forward(self, x, is_train=True):
        flag_clip = 1
        if is_train:
            message = self.train_message
        else:
            message = self.test_message
        odd_result = self.odd_layer(x, message, self.oddw_v1, self.oddw_e1)
        even_result1 = self.even_layer(odd_result, flag_clip)

        flag_clip = 0
        odd_result = self.odd_layer(
            x, even_result1, self.oddw_v2, self.oddw_e2)
        even_result2 = self.even_layer(odd_result, flag_clip)

        odd_result = self.odd_layer(
            x, even_result2, self.oddw_v3, self.oddw_e3)
        even_result3 = self.even_layer(odd_result, flag_clip)

        odd_result = self.odd_layer(
            x, even_result3, self.oddw_v4, self.oddw_e4)
        even_result4 = self.even_layer(odd_result, flag_clip)

        odd_result = self.odd_layer(
            x, even_result4, self.oddw_v5, self.oddw_e5)
        even_result5 = self.even_layer(odd_result, flag_clip)

        output = self.output_layer(x, even_result5)

        return output
