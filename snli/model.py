import torch
from torch import nn
from torch.nn import init

from model.treelstm import BinaryTreeLSTM


class SNLIClassifier(nn.Module):

    def __init__(self, num_classes, input_dim, hidden_dim, num_layers,
                 use_batchnorm, dropout_prob):
        super(SNLIClassifier, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob

        if use_batchnorm:
            self.bn_mlp_input = nn.BatchNorm1d(num_features=4 * input_dim)
            self.bn_mlp_output = nn.BatchNorm1d(num_features=hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        mlp_layers = []
        for i in range(num_layers):
            layer_in_features = hidden_dim if i > 0 else 4 * input_dim
            linear_layer = nn.Linear(in_features=layer_in_features,
                                     out_features=hidden_dim)
            relu_layer = nn.ReLU()
            mlp_layer = nn.Sequential(linear_layer, relu_layer)
            mlp_layers.append(mlp_layer)
        self.mlp = nn.Sequential(*mlp_layers)
        self.clf_linear = nn.Linear(in_features=hidden_dim,
                                    out_features=num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_batchnorm:
            self.bn_mlp_input.reset_parameters()
            self.bn_mlp_output.reset_parameters()
        for i in range(self.num_layers):
            linear_layer = self.mlp[i][0]
            init.kaiming_normal(linear_layer.weight.data)
            init.constant(linear_layer.bias.data, val=0)
        init.uniform(self.clf_linear.weight.data, -0.005, 0.005)
        init.constant(self.clf_linear.bias.data, val=0)

    def forward(self, pre, hyp):
        f1 = pre
        f2 = hyp
        f3 = torch.abs(pre - hyp)
        f4 = pre * hyp
        mlp_input = torch.cat([f1, f2, f3, f4], dim=1)
        if self.use_batchnorm:
            mlp_input = self.bn_mlp_input(mlp_input)
        mlp_input = self.dropout(mlp_input)
        mlp_output = self.mlp(mlp_input)
        if self.use_batchnorm:
            mlp_output = self.bn_mlp_output(mlp_output)
        mlp_output = self.dropout(mlp_output)
        logits = self.clf_linear(mlp_output)
        return logits


class SNLIModel(nn.Module):

    def __init__(self, num_classes, num_words, word_dim, hidden_dim,
                 clf_hidden_dim, clf_num_layers, use_leaf_rnn, intra_attention,
                 use_batchnorm, dropout_prob):
        super(SNLIModel, self).__init__()
        self.num_classes = num_classes
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.clf_hidden_dim = clf_hidden_dim
        self.clf_num_layers = clf_num_layers
        self.use_leaf_rnn = use_leaf_rnn
        self.intra_attention = intra_attention
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob

        self.word_embedding = nn.Embedding(num_embeddings=num_words,
                                           embedding_dim=word_dim)
        self.encoder = BinaryTreeLSTM(word_dim=word_dim, hidden_dim=hidden_dim,
                                      use_leaf_rnn=use_leaf_rnn,
                                      intra_attention=intra_attention,
                                      gumbel_temperature=1)
        self.classifier = SNLIClassifier(
            num_classes=num_classes, input_dim=hidden_dim,
            hidden_dim=clf_hidden_dim, num_layers=clf_num_layers,
            use_batchnorm=use_batchnorm, dropout_prob=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.reset_parameters()

    def reset_parameters(self):
        init.normal(self.word_embedding.weight.data, mean=0, std=0.01)
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, pre, pre_length, hyp, hyp_length):
        pre_embeddings = self.word_embedding(pre)
        hyp_embeddings = self.word_embedding(hyp)
        pre_embeddings = self.dropout(pre_embeddings)
        hyp_embeddings = self.dropout(hyp_embeddings)
        pre_h, _ = self.encoder(input=pre_embeddings, length=pre_length)
        hyp_h, _ = self.encoder(input=hyp_embeddings, length=hyp_length)
        logits = self.classifier(pre=pre_h, hyp=hyp_h)
        return logits
