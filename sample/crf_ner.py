from torch import nn
from torch.nn import Dropout, Conv1d
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from transformers import Conv1D

from core.abstract_classifier import AbstractClassifier

# WIP ...
class CRFNERModel(nn.Module):
    def __init__(self, vocab_size: int):
        super(CRFNERModel, self).__init__()

        # self.char_emb = nn.Embedding(128, 30)
        # self.char_dropout = Dropout()
        self.word_emb = nn.Embedding(vocab_size, 300)

        hidden_dim = 20
        self.lstm = nn.LSTM(300, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)


    def forward(self, x):
        y = nn.functional.relu(self.fc(x))
        return y


class CRFNER(AbstractClassifier):

    def __init__(self, input_size: int):
        super(CRFNER, self).__init__()
        self.input_size = input_size

    def get_model(self, num_classes: int, **kwargs):
        self.model = CRFNERModel(self.input_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), betas=(0.9, 0.98), eps=10e-9)
        return self.model, self.criterion, self.optimizer

    def load_data(self) -> (DataLoader, {}):
        label2idx = {"yes": 1, "no": 0}

        input_size = self.input_size

        class SampleDataset(Dataset):
            def __init__(self):
                self.count = 0

            def __len__(self):
                return 1000

            def __getitem__(self, idx):
                start = min(label2idx.values())
                end = max(label2idx.values()) + 1
                y = np.random.randint(start, end)

                self.count += 1
                return torch.rand(input_size), y, f"id_{self.count}"

        ds = SampleDataset()
        return DataLoader(ds, batch_size=1000), label2idx











        # Make up some training data
        training_data = [(
            "the wall street journal reported today that apple corporation made money".split(),
            "B I I I O O O B I O O".split()
        ), (
            "georgia tech is a university in georgia".split(),
            "B I O O O O B".split()
        )]











# char
        char_input = Input(shape=(None, char_maxlen,), name="char_input")
        # not maintain input dim, directly using ASCII
        char_layer = TimeDistributed(
            Embedding(input_dim=128, output_dim=30, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), trainable=True, name="char_emb")(char_input)
        char_layer = Dropout(0.5, name="char_dropout_1")(char_layer)
        char_layer = TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same', activation='tanh', strides=1), name="char_conv")(char_layer)





        char_layer = TimeDistributed(MaxPooling1D(char_maxlen), name="char_maxpool")(char_layer)
        char_layer = TimeDistributed(Flatten(), name="char_flatten")(char_layer)
        char_layer = Dropout(0.5, name="char_dropout_2")(char_layer)

        # word
        words_input = Input(shape=(None,), dtype="int32", name="words_input")

        if word_emb is not None:  # emb_matrix.shape=vocab_size+1, emb_dim("50")
            words_layer = Embedding(input_dim=word_emb.shape[0], output_dim=word_emb.shape[1], weights=[word_emb], mask_zero=True, trainable=False, name="word_emb")(words_input)
        else:
            words_layer = Embedding(input_dim=vocab_size + 1, output_dim=emb_dim, mask_zero=True, trainable=False, name="word_emb")(words_input)

        feat_input = Input(shape=(None, feat_size), dtype="float32", name="feat_input")
        feat_layer = Dense(input_dim=feat_size, activation="relu", units=feat_size, trainable=True)(feat_input)
        shared_output = concatenate([words_layer, char_layer, feat_layer], name="concat")

        shared_output = Bidirectional(LSTM(bi_rnn_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.25), name="BiLSTM_1")(shared_output)
        shared_output = MultiHeadAttention(head_num=multi_head_num, name="Multihead_Attn")(shared_output)
        shared_output = BatchNormalization(name="BatchNorm1")(shared_output)

        intent_output = Bidirectional(LSTM(bi_rnn_units, return_sequences=False, dropout=0.5, recurrent_dropout=0.25), name="Intent_BiLSTM")(shared_output)
        intent_output = BatchNormalization(name="Intent_BatchNorm2")(intent_output)
        intent_output = Dense(label_size[0], activation="softmax", name="Intent_softmax")(intent_output)

        # had to set Embedding mask_zero=True for avoid neg loss. https://github.com/keras-team/keras-contrib/issues/278
        bi_crf = CRF(label_size[1], sparse_target=True, name="BI_CRF")
        bi_output = bi_crf(shared_output)

        concept_crf = CRF(label_size[2], sparse_target=True, name="Concept_CRF")
        concept_output = concept_crf(shared_output)

        emd_crf = CRF(label_size[3], sparse_target=True, name="EMD_CRF")
        emd_output = emd_crf(shared_output)

        model = Model(inputs=[words_input, char_input, feat_input], outputs=[intent_output, bi_output, concept_output, emd_output])
        adam_optimizer = Adam(beta_1=0.9, beta_2=0.98, epsilon=10e-9)

        model.compile(optimizer=adam_optimizer, loss=["categorical_crossentropy", crf_loss_joined(bi_crf), crf_loss_joined(concept_crf), crf_loss_joined(emd_crf)],
                      metrics={"Intent_softmax": "accuracy", "BI_CRF": crf_accuracy_joined(bi_crf), "Concept_CRF": crf_accuracy_joined(concept_crf),
                               "EMD_CRF": crf_accuracy_joined(emd_crf)})