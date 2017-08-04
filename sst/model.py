from torch import nn
from torch.nn import init

from model.treelstm import BinaryTreeLSTM


class SSTClassifier(nn.Module):

    def __init__(self, num_classes, input_dim, hidden_dim, num_layers,
                 use_batchnorm, dropout_prob):
        super(SSTClassifier, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob

        if use_batchnorm:
            self.bn_mlp_input = nn.BatchNorm1d(num_features=input_dim)
            self.bn_mlp_output = nn.BatchNorm1d(num_features=hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        mlp_layers = []
        for i in range(num_layers):
            layer_in_features = hidden_dim if i > 0 else input_dim
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
        init.kaiming_normal(self.clf_linear.weight.data)
        init.constant(self.clf_linear.bias.data, val=0)

    def forward(self, sentence):
        mlp_input = sentence
        if self.use_batchnorm:
            mlp_input = self.bn_mlp_input(mlp_input)
        mlp_input = self.dropout(mlp_input)
        mlp_output = self.mlp(mlp_input)
        if self.use_batchnorm:
            mlp_output = self.bn_mlp_output(mlp_output)
        mlp_output = self.dropout(mlp_output)
        logits = self.clf_linear(mlp_output)
        return logits


class SSTModel(nn.Module):

    def __init__(self, num_classes, num_words, word_dim, hidden_dim,
                 clf_hidden_dim, clf_num_layers, use_leaf_rnn, use_leaf_birnn,
                 intra_attention, use_batchnorm, dropout_prob):
        super(SSTModel, self).__init__()
        self.num_classes = num_classes
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.clf_hidden_dim = clf_hidden_dim
        self.clf_num_layers = clf_num_layers
        self.use_leaf_rnn = use_leaf_rnn
        self.use_leaf_birnn = use_leaf_birnn
        self.intra_attention = intra_attention
        self.use_batchnorm = use_batchnorm
        self.dropout_prob = dropout_prob

        self.dropout = nn.Dropout(dropout_prob)
        self.word_embedding = nn.Embedding(num_embeddings=num_words,
                                           embedding_dim=word_dim)
        self.encoder = BinaryTreeLSTM(word_dim=word_dim, hidden_dim=hidden_dim,
                                      use_leaf_rnn=use_leaf_rnn,
                                      use_leaf_birnn=use_leaf_birnn,
                                      intra_attention=intra_attention,
                                      gumbel_temperature=1)
        self.classifier = SSTClassifier(
            num_classes=num_classes, input_dim=hidden_dim,
            hidden_dim=clf_hidden_dim, num_layers=clf_num_layers,
            use_batchnorm=use_batchnorm, dropout_prob=dropout_prob)
        self.reset_parameters()

    def reset_parameters(self):
        init.normal(self.word_embedding.weight.data, mean=0, std=0.01)
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, words, length):
        words_embed = self.word_embedding(words)
        words_embed = self.dropout(words_embed)
        sentence_vector, _ = self.encoder(input=words_embed, length=length)
        logits = self.classifier(sentence_vector)
        return logits
