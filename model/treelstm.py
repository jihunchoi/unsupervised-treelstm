import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import init

from . import basic


class BinaryTreeLSTMLayer(nn.Module):

    def __init__(self, hidden_dim):
        super(BinaryTreeLSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.comp_linear = nn.Linear(in_features=2 * hidden_dim,
                                     out_features=5 * hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal(self.comp_linear.weight.data)
        init.constant(self.comp_linear.bias.data, val=0)

    def forward(self, l=None, r=None):
        """
        Args:
            l: A (h_l, c_l) tuple, where each value has the size
                (batch_size, max_length, hidden_dim).
            r: A (h_r, c_r) tuple, where each value has the size
                (batch_size, max_length, hidden_dim).
        Returns:
            h, c: The hidden and cell state of the composed parent,
                each of which has the size
                (batch_size, max_length - 1, hidden_dim).
        """

        hl, cl = l
        hr, cr = r
        hlr_cat = torch.cat([hl, hr], dim=2)
        treelstm_vector = basic.apply_nd(fn=self.comp_linear, input=hlr_cat)
        i, fl, fr, u, o = treelstm_vector.chunk(num_chunks=5, dim=2)
        c = (cl*(fl + 1).sigmoid() + cr*(fr + 1).sigmoid()
             + u.tanh()*i.sigmoid())
        h = o.sigmoid() * c.tanh()
        return h, c


class BinaryTreeLSTM(nn.Module):

    def __init__(self, word_dim, hidden_dim, gumbel_temperature):
        super(BinaryTreeLSTM, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.gumbel_temperature = gumbel_temperature

        self.word_linear = nn.Linear(in_features=word_dim,
                                     out_features=2 * hidden_dim)
        self.treelstm_layer = BinaryTreeLSTMLayer(hidden_dim)
        self.comp_query = nn.Parameter(torch.FloatTensor(hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal(self.word_linear.weight.data)
        init.constant(self.word_linear.bias.data, val=0)
        self.treelstm_layer.reset_parameters()
        init.normal(self.comp_query.data, mean=0, std=0.01)

    @staticmethod
    def update_state(old_state, new_state, depth, length):
        old_h, old_c = old_state
        new_h, new_c = new_state
        mask = (torch.gt(length, depth).float().unsqueeze(1).unsqueeze(2)
                .expand_as(new_h))
        h = mask * new_h + (1 - mask) * old_h[:, :-1, :]
        c = mask * new_c + (1 - mask) * old_c[:, :-1, :]
        return h, c

    def select_composition(self, old_state, new_state):
        new_h, new_c = new_state
        old_h, old_c = old_state
        old_h_left, old_h_right = old_h[:, :-1, :], old_h[:, 1:, :]
        old_c_left, old_c_right = old_c[:, :-1, :], old_c[:, 1:, :]
        comp_weights = basic.dot_nd(query=self.comp_query, candidates=new_h)
        if self.training:
            select_mask = basic.st_gumbel_softmax(
                logits=comp_weights, temperature=self.gumbel_temperature)
        else:
            select_mask = basic.convert_to_one_hot(
                indices=comp_weights.max(1)[1].squeeze(1),
                num_classes=comp_weights.size(1))
            select_mask = select_mask.float()
        select_mask_expand = select_mask.unsqueeze(2).expand_as(new_h)
        select_mask_cumsum = select_mask.cumsum(1)
        left_mask = 1 - select_mask_cumsum
        left_mask_expand = left_mask.unsqueeze(2).expand_as(old_h_left)
        right_mask_leftmost_col = Variable(
            select_mask_cumsum.data.new(new_h.size(0), 1).zero_())
        right_mask = torch.cat(
            [right_mask_leftmost_col, select_mask_cumsum[:, :-1]], dim=1)
        right_mask_expand = right_mask.unsqueeze(2).expand_as(old_h_right)
        new_h = (select_mask_expand * new_h
                 + left_mask_expand * old_h_left
                 + right_mask_expand * old_h_right)
        new_c = (select_mask_expand * new_c
                 + left_mask_expand * old_c_left
                 + right_mask_expand * old_c_right)
        return new_h, new_c

    def forward(self, input, length):
        max_depth = input.size(1)

        state = basic.apply_nd(fn=self.word_linear, input=input)
        state = state.chunk(num_chunks=2, dim=2)
        for i in range(max_depth - 1):
            h, c = state
            l = (h[:, :-1, :], c[:, :-1, :])
            r = (h[:, 1:, :], c[:, 1:, :])
            new_state = self.treelstm_layer(l=l, r=r)
            if i < max_depth - 2:
                # We don't need to greedily select the composition in the
                # last iteration, since it has only one option left.
                new_state = self.select_composition(old_state=state,
                                                    new_state=new_state)
            state = self.update_state(old_state=state, new_state=new_state,
                                      depth=i, length=length)
        h, c = state
        assert h.size(1) == 1 and c.size(1) == 1
        return h.squeeze(1), c.squeeze(1)
