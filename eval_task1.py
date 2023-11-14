import warnings
warnings.filterwarnings('ignore')

import datasets
import torch
import itertools
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset
from conlleval import evaluate
from collections import Counter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#creating dataset to be trained in batch

dataset_train = datasets.load_dataset("conll2003", split = 'train')
dataset_test = datasets.load_dataset("conll2003", split = 'test')
dataset_dev = datasets.load_dataset("conll2003", split = 'validation')

word_freq = (Counter(itertools.chain(*dataset_train['tokens'])))

#word in tokens for each row in dataset becomes the key and value is the freq
word_freq = {
    word : freq
    for word, freq in word_freq.items() if freq >= 3
}

word2idx = {
    word: index
    for index, word in enumerate(word_freq.keys(), start = 2)
}

word2idx['[PAD]'] = 0
word2idx['[UNK]'] = 1

# Rename 'ner_tags' to 'labels' in each split
dataset_train = dataset_train.rename_column('ner_tags', 'labels')
dataset_test = dataset_test.rename_column('ner_tags', 'labels')
dataset_dev = dataset_dev.rename_column('ner_tags', 'labels')


def get_padded_input_ids_and_labels_train(df):
    input_ids = [[word2idx.get(token, word2idx['[UNK]']) for token in sample] for sample in df['tokens']]
    input_ids = [torch.tensor(input_ids_sample) for input_ids_sample in input_ids]
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

    max_seq_len = max(len(seq) for seq in input_ids)
    label_tensors_data = [torch.tensor(seq + [9]*(max_seq_len-len(seq))) for seq in df['labels']]
    padded_labels_data = pad_sequence(label_tensors_data, batch_first=True, padding_value=9)

    data_new = TensorDataset(padded_input_ids, padded_labels_data)
    data_loader_ = DataLoader(data_new, batch_size=32, shuffle=True)
    return data_loader_, padded_input_ids, max_seq_len


def get_padded_input_ids_and_labels(df, max_seq_len):
    input_ids = [[word2idx.get(token, word2idx['[UNK]']) for token in sample] for sample in df['tokens']]
    input_ids = [torch.tensor(input_ids_sample) for input_ids_sample in input_ids]
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    label_tensors_data = [torch.tensor(seq + [9]*(max_seq_len-len(seq))) for seq in df['labels']]
    padded_labels_data = pad_sequence(label_tensors_data, batch_first=True, padding_value=9)

    data_new = TensorDataset(padded_input_ids, padded_labels_data)
    data_loader_ = DataLoader(data_new, batch_size=32, shuffle=True)
    return data_loader_, padded_input_ids


padded_dataset_train, train_input_ids, max_seq_len_train = get_padded_input_ids_and_labels_train(dataset_train)
padded_dataset_dev, dev_input_ids = get_padded_input_ids_and_labels(dataset_dev, max_seq_len_train)
padded_dataset_test, test_input_ids = get_padded_input_ids_and_labels(dataset_test, max_seq_len_train)


class BiLSTM(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout, num_labels):
    super(BiLSTM, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
    self.dropout = nn.Dropout()
    self.linear = nn.Linear(2*hidden_dim, output_dim)
    self.elu = nn.ELU()
    self.classifier = nn.Linear(output_dim, num_labels)

  def forward(self, input):
    embeddings = (self.embedding(input))
    x, _ = self.lstm(embeddings)
    x = self.dropout(x)
    x = self.linear(x)
    x = self.elu(x)
    logits = self.classifier(x)
    return logits


# Define hyperparameters
embedding_dim = 100
hidden_dim = 256
output_dim = 128
num_layers = 1
dropout = 0.33
vocab_size = len(word2idx)  # Adjust based on your vocabulary size
num_epochs = 20
initial_lr = 0.01
num_labels = 9

model = BiLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout, num_labels)
model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=9)
optimizer = optim.AdamW(model.parameters(), lr=initial_lr)

step_size = 5
scheduler = StepLR(optimizer=optimizer, step_size=step_size, gamma = 0.1)

model.load_state_dict(torch.load('bilstm1_state_dict.pt', map_location=device))
model.eval()


def remove_padding(preds, unpadded_labels):
    unpadded_preds = []
    for i in range(len(unpadded_labels)):
        unpadded_preds.append(preds[i][:len(unpadded_labels[i])])
    unpadded_preds = [pred.tolist() for pred in unpadded_preds]

    return unpadded_preds


#input tokens are padded
#dataset_dev
def evaluate_model(model, padded_input_tokens, dataset):
    model.eval()

    with torch.no_grad():
        padded_input_tokens = padded_input_tokens.to(device)
        predictions = model(padded_input_tokens)
        predictions = torch.argmax(predictions, dim=-1)

    unpadded_preds = remove_padding(predictions, dataset['labels'])


    idx2tag = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
    labels = [list(map(idx2tag.get, labels)) for labels in dataset['labels']]
    preds_string = [list(map(idx2tag.get, labels)) for labels in unpadded_preds]

    precision, recall, f1 = evaluate(itertools.chain(*labels), itertools.chain(*preds_string))

    return precision, recall, f1

precision, recall, f1 = evaluate_model(model, test_input_ids, dataset_test)
