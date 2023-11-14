'''
    HW4 CSCI-544
    Ananya Kotha
    7427344242
    Python version: 3.11.6
'''


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
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_train = datasets.load_dataset("conll2003", split = 'train')
dataset_test = datasets.load_dataset("conll2003", split = 'test')
dataset_dev = datasets.load_dataset("conll2003", split = 'validation')

# Rename 'ner_tags' to 'labels' in each split
dataset_train = dataset_train.rename_column('ner_tags', 'labels')
dataset_test = dataset_test.rename_column('ner_tags', 'labels')
dataset_dev = dataset_dev.rename_column('ner_tags', 'labels')

glove_word2idx = {}
glove_word2idx['[PAD]'] = 0
glove_word2idx['[UNK]'] = 1

glove_vocab, glove_embeddings = [], []
with open('glove.6B.100d.txt', 'rt', encoding='utf-8') as f:
    all_file_embeddings = f.read().strip().split('\n')

for i in range(len(all_file_embeddings)):
    glove_word = all_file_embeddings[i].split(' ')[0]  #tokenizing
    glove_embed = [float(x) for x in all_file_embeddings[i].split(' ')[1:]]  #read or store each embedding as a float
    glove_word2idx[glove_word] = i+2
    glove_vocab.append(glove_word)
    glove_embeddings.append(glove_embed)


glove_vocab_npa = np.array(glove_vocab)
glove_embeddings_npa = np.array(glove_embeddings)


#insert '<pad>' and '<unk>' tokens at start of vocab_npa.
glove_vocab_npa = np.insert(glove_vocab_npa, 0, '[PAD]')
glove_vocab_npa = np.insert(glove_vocab_npa, 1, '[UNK]')
#embedding for '<pad>' token
pad_emb_npa = np.zeros((1,glove_embeddings_npa.shape[1]))
# embedding for '<unk>' token
unk_emb_npa = np.mean(glove_embeddings_npa,axis=0,keepdims=True)
#insert embeddings for pad and unk tokens at top of embs_npa.
glove_embeddings_npa = np.vstack((pad_emb_npa,unk_emb_npa,glove_embeddings_npa))

my_embedding_layer = torch.nn.Embedding.from_pretrained(torch.from_numpy(glove_embeddings_npa).float(), freeze = True, padding_idx=0)


max_seq_length_train_glove = max(len(sequence) for sequence in dataset_train['tokens'])
max_seq_length_test_glove = max(len(sequence) for sequence in dataset_test['tokens'])
max_seq_length_dev_glove = max(len(sequence) for sequence in dataset_dev['tokens'])


def find_additional_features(data, max_sequence_length):
    features_data = []
    for sequence in data:
        add_features = [[float(token.istitle()), float(token.isupper()), float(token.islower())] for token in sequence]
        add_features += [[0.0, 0.0, 0.0]] * (max_sequence_length - len(add_features))
        features_data.append(add_features)
    return torch.tensor(features_data)


addn_features = find_additional_features(dataset_train['tokens'],max_seq_length_train_glove)


def get_padded_input_ids_and_labels_glove(df, add_feats, max_seq_len):
    input_ids = [[glove_word2idx.get(token.lower(), glove_word2idx['[UNK]']) for token in sample] for sample in df['tokens']]
    input_ids = [torch.tensor(input_ids_sample) for input_ids_sample in input_ids]
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

    label_tensors = [torch.tensor(seq + [9]*(max_seq_len-len(seq))) for seq in df['labels']]
    padded_labels = pad_sequence(label_tensors, batch_first=True, padding_value=9)

    data = TensorDataset(padded_input_ids, add_feats, padded_labels)
    data_loader = DataLoader(data, batch_size=32, shuffle=True)
    return data_loader, padded_input_ids


#Embedding → BiLSTM → Linear → ELU → classifier
class BiLSTM_glove(nn.Module):
    def __init__(self, embedding_dim, hidden_dim,
        output_dim, num_layers, dropout, num_labels):
        super(BiLSTM_glove, self).__init__()
        self.embedding = my_embedding_layer
        self.lstm = nn.LSTM(embedding_dim+3, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.33)
        self.linear = nn.Linear(2*hidden_dim, output_dim)
        self.elu = nn.ELU()
        self.classifier = nn.Linear(output_dim, num_labels)

    def forward(self, input, add_features):
        embeddings = (self.embedding(input))
        add_features = add_features.to(embeddings.device)
        embeddings_with_add_features = torch.cat((embeddings, add_features), dim= 2)
        x, _ = self.lstm(embeddings_with_add_features)
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
num_labels = 9
num_epochs = 20
initial_lr = 0.01

model_glove = BiLSTM_glove(embedding_dim, hidden_dim, output_dim, num_layers, dropout, num_labels)
model_glove.to(device)

criterion_glove = nn.CrossEntropyLoss(ignore_index=9)
optimizer_glove = optim.AdamW(model_glove.parameters(), lr=initial_lr)

step_size_glove = 5
scheduler_glove = StepLR(optimizer=optimizer_glove, step_size=step_size_glove, gamma=0.1)

model_glove.load_state_dict(torch.load('bilstm2_state_dict.pt', map_location=device))
model_glove.eval()

dev_addn_features = find_additional_features(dataset_dev['tokens'], max_seq_length_dev_glove)
test_addn_features = find_additional_features(dataset_test['tokens'], max_seq_length_test_glove)

dev_data_loader_glove, dev_input_ids_glove = get_padded_input_ids_and_labels_glove(dataset_dev, dev_addn_features,
                                                                                   max_seq_length_dev_glove)
test_data_loader_glove, test_input_ids_glove = get_padded_input_ids_and_labels_glove(dataset_test, test_addn_features,
                                                                                     max_seq_length_test_glove)


def remove_padding(preds, unpadded_labels):
    unpadded_preds = []
    for i in range(len(unpadded_labels)):
        unpadded_preds.append(preds[i][:len(unpadded_labels[i])])

    unpadded_preds = [pred.tolist() for pred in unpadded_preds]

    return unpadded_preds


# input tokens are padded
# dataset_dev
def evaluate_model_glove(model, padded_input_tokens, dataset, add_features):
    model.eval()

    with torch.no_grad():
        padded_input_tokens = padded_input_tokens.to(device)
        predictions = model(padded_input_tokens, add_features)
        predictions = torch.argmax(predictions, dim=-1)

    unpadded_preds = remove_padding(predictions, dataset['labels'])

    idx2tag = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'}
    labels = [list(map(idx2tag.get, labels)) for labels in dataset['labels']]
    preds_string = [list(map(idx2tag.get, labels)) for labels in unpadded_preds]

    precision, recall, f1 = evaluate(itertools.chain(*labels), itertools.chain(*preds_string))

    return precision, recall, f1

precision, recall, f1 = evaluate_model_glove(model_glove, test_input_ids_glove, dataset_test, test_addn_features)
