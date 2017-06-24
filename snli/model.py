import torch
from torch import nn
from torch.nn import init

from model.treelstm import BinaryTreeLSTM


class SNLIClassifier(nn.Module):

    def __init__(self, num_classes, input_dim, hidden_dim, num_layers):
        super(SNLIClassifier, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        mlp_layers = []
        for i in range(num_layers + 1):
            layer_in_features = hidden_dim if i > 0 else 4 * input_dim
            layer_out_features = hidden_dim if i < num_layers else num_classes
            linear_layer = nn.Linear(in_features=layer_in_features,
                                     out_features=layer_out_features)
            if i < num_layers:
                relu_layer = nn.ReLU()
                mlp_layer = nn.Sequential(linear_layer, relu_layer)
            else:
                mlp_layer = nn.Sequential(linear_layer)
            mlp_layers.append(mlp_layer)
        self.mlp = nn.Sequential(*mlp_layers)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_layers):
            linear_layer = self.mlp[i][0]
            init.kaiming_normal(linear_layer.weight.data)
            init.constant(linear_layer.bias.data, val=0)
        last_linear_layer = self.mlp[self.num_layers][0]
        init.uniform(last_linear_layer.weight.data, -0.005, 0.005)
        init.constant(last_linear_layer.bias.data, val=0)

    def forward(self, pre, hyp):
        f1 = pre
        f2 = hyp
        f3 = pre - hyp
        f4 = pre * hyp
        mlp_input = torch.cat([f1, f2, f3, f4], dim=1)
        logits = self.mlp(mlp_input)
        return logits


class SNLIModel(nn.Module):

    def __init__(self, num_classes, num_words, word_dim, hidden_dim,
                 clf_hidden_dim, clf_num_layers, use_leaf_rnn):
        super(SNLIModel, self).__init__()
        self.num_classes = num_classes
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.clf_hidden_dim = clf_hidden_dim
        self.clf_num_layers = clf_num_layers
        self.use_leaf_rnn = use_leaf_rnn

        self.word_embedding = nn.Embedding(num_embeddings=num_words,
                                           embedding_dim=word_dim)
        self.encoder = BinaryTreeLSTM(word_dim=word_dim, hidden_dim=hidden_dim,
                                      use_leaf_rnn=use_leaf_rnn,
                                      gumbel_temperature=1)
        self.classifier = SNLIClassifier(
            num_classes=num_classes, input_dim=hidden_dim,
            hidden_dim=clf_hidden_dim, num_layers=clf_num_layers)
        self.reset_parameters()

    def reset_parameters(self):
        init.normal(self.word_embedding.weight.data, mean=0, std=0.01)
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, pre, pre_length, hyp, hyp_length):
        pre_embeddings = self.word_embedding(pre)
        hyp_embeddings = self.word_embedding(hyp)
        pre_h, _ = self.encoder(input=pre_embeddings, length=pre_length)
        hyp_h, _ = self.encoder(input=hyp_embeddings, length=hyp_length)
        logits = self.classifier(pre=pre_h, hyp=hyp_h)
        return logits
