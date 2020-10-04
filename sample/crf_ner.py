import datasets
from datasets import load_dataset
from torch import nn, IntTensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

from core.abstract_classifier import AbstractClassifier

# WIP ...
from util.utils import flatten_list


class CRFNERModel(nn.Module):
    def __init__(self, num_classes: int):
        super(CRFNERModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = BertModel.from_pretrained('bert-base-cased')
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=1e-5)

    def forward(self, x):
        # def forward(
        #         self,
        #         input_ids=None,
        #         attention_mask=None,
        #         token_type_ids=None,
        #         position_ids=None,
        #         head_mask=None,
        #         inputs_embeds=None,
        #         encoder_hidden_states=None,
        #         encoder_attention_mask=None,
        #         output_attentions=None,
        #         output_hidden_states=None,
        #         return_dict=None,
        # ):
        y = self.model.forward(**x)
        last_hidden_states = y.last_hidden_state
        return last_hidden_states


class CRFNER(AbstractClassifier):
    def __init__(self):
        super(CRFNER, self).__init__()

    def get_model(self, num_classes: int, **kwargs):
        self.model = CRFNERModel(num_classes)
        return self.model, self.model.criterion, self.model.optimizer

    def load_data(self, data, tokenizer) -> (DataLoader, {}):
        self.label2idx = {"[CLS]": 0, "[SEP]": 1, "[PAD]": 2}

        class SeqDataset(Dataset):
            def __init__(self, data: {}):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                x = self.data[idx]["sentence"]
                y = self.data[idx]["label"]
                return x, y, f"id_{idx}"

        max_len = 0
        for it in data:
            raw_text_arr = it[0]
            label_arr = it[1]
            if len(raw_text_arr) != len(label_arr):
                continue
            max_len = max(max_len, len(label_arr))

        transformed_datasets = []
        for idx, it in enumerate(data):
            raw_text_arr = it[0]
            label_arr = it[1]

            if len(label_arr) < max_len:
                for _ in range(max_len - len(label_arr)):
                    label_arr.append("[PAD]")
                    raw_text_arr.append("[PAD]")

            y = [0]  # CLS
            for label in label_arr:
                if label not in self.label2idx.keys():
                    self.label2idx[label] = len(self.label2idx)
                y.append(self.label2idx[label])

            y.append(1)  # SEP
            y = IntTensor(y)
            sentence = " ".join(raw_text_arr)  # TODO: hack

            if len(raw_text_arr) != len(y) - 2:
                print(f"unable to load {' '.join(raw_text_arr)}. len(tokens): {len(raw_text_arr)} len(labels): {len(y) - 2}")

            transformed_datasets.append({"sentence": sentence, "label": y, "idx": idx})

        def encode(examples):
            return tokenizer(examples['sentence'], truncation=True, padding='max_length')

        # dataset = load_dataset('csv', data_files='my_file.csv')
        dataset = datasets.arrow_dataset.Dataset.from_dict({"sentence": "aa", "label": "aa", "idx": 1})
        dataset = dataset.map(encode, batched=True)
        return DataLoader(dataset, batch_size=1000), self.label2idx


if __name__ == "__main__":
    cf = CRFNER()
    model, criterion, optimizer = cf.get_model(2)
    # dataset = load_dataset('json', data_files="/Users/castor/Desktop/a.json")
    dataset = load_dataset("polyglot_ner", "en")
    dataset = dataset["train"][0:2]
    dataset = datasets.Dataset.from_dict(dataset)
    tokenizer = model.tokenizer


    def encode(data):
        return tokenizer(text=data['words'], is_split_into_words=True, padding='max_length')


    dataset = dataset.map(encode, batched=True)
    # dataset = dataset.map(lambda it: {'labels': it['label']}, batched=True)
    labels = set(flatten_list(dataset["ner"]))
    cf.label2idx = {label: idx + 3 for idx, label in enumerate(labels)}
    cf.label2idx.update({"[CLS]": 0, "[SEP]": 1, "[PAD]": 2})
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'ner'])
    dl = DataLoader(dataset, batch_size=32)
    cf.predict(dl)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    # model = BertModel.from_pretrained('bert-base-cased')
    #
    # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # outputs = model(**inputs)
    #
    # print(outputs)
