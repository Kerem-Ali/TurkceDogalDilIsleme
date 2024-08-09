#!pip install datasets==2.0.1
from datasets import load_dataset, load_from_disk
import pandas as pd


# Later, load the dataset from disk
dataset = load_dataset("wikiann", "tr")

# Verify that the dataset is loaded correctly
ner_encoding = {0: "O", 1: "entity", 2: "entity", 3: "entity", 4: "entity", 5: "entity", 6: "entity"}


train_tokens = []
train_tags = []
for sample in dataset["train"]:
  train_tokens.append(' '.join(sample["tokens"]))
  train_tags.append(' '.join([ner_encoding[a] for a in sample["ner_tags"]]))

test_tokens = []
test_tags = []
for sample in dataset["train"]:
  test_tokens.append(' '.join(sample["tokens"]))
  test_tags.append(' '.join([ner_encoding[a] for a in sample["ner_tags"]]))

df_train = pd.DataFrame({"sentence": train_tokens, "tags": train_tags})
df_test = pd.DataFrame({"sentence": test_tokens, "tags": test_tags})


ner_encoding = {0: "O", 1: "entity", 2: "entity", 3: "entity", 4: "entity", 5: "entity", 6: "entity"}

texts = []
for sample in dataset["train"]:
  texts.append(' '.join(sample["tokens"]))
for sample in dataset["test"]:
  texts.append(' '.join(sample["tokens"]))

labels = []
for sample in dataset["train"]:
  labels.append([ner_encoding[a] for a in sample["ner_tags"]])
for sample in dataset["test"]:
  labels.append([ner_encoding[a] for a in sample["ner_tags"]])



import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchcrf import CRF

# Define NERDataset class
class NERDataset(Dataset):
    def __init__(self, texts, labels, vocab, tag_to_idx):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tag_to_idx = tag_to_idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = [self.vocab.get(word, self.vocab['<UNK>']) for word in self.texts[idx].split()]
        labels = [self.tag_to_idx[tag] for tag in self.labels[idx]]
        return torch.tensor(text, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

# Define BiLSTM_CRF model class
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_idx, embedding_dim=200, hidden_dim=256):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_idx = tag_to_idx
        self.tagset_size = len(tag_to_idx)

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=2, bidirectional=True, dropout=0.5)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)

    def forward(self, sentence, tags=None):
        embedded = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embedded)
        tag_scores = self.hidden2tag(lstm_out)

        if tags is not None:
            mask = (sentence != 0).bool()
            loss = -self.crf(tag_scores, tags, mask=mask)
            return loss
        else:
            best_tags = self.crf.decode(tag_scores)
            return best_tags

# Prepare data
def prepare_data(texts, labels):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    tag_to_idx = {'O': 0, 'entity': 1}

    for text in texts:
        for word in text.split():
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab, tag_to_idx

# Collate function for DataLoader
def collate_fn(batch):
    texts, labels = zip(*batch)
    max_len = max(len(x) for x in texts)
    padded_texts = [torch.cat([x, torch.zeros(max_len - len(x), dtype=torch.long)]) for x in texts]
    padded_labels = [torch.cat([y, torch.zeros(max_len - len(y), dtype=torch.long)]) for y in labels]
    return torch.stack(padded_texts), torch.stack(padded_labels)

# Example data (texts and labels should be extended with more diverse data)


# Check and align lengths of texts and labels
for i in range(len(texts)):
    text_length = len(texts[i].split())
    label_length = len(labels[i])
    if text_length != label_length:
        raise ValueError(f"Text and label lengths do not match at index {i}: {text_length} != {label_length}")

# Prepare vocabulary and tag_to_idx
vocab, tag_to_idx = prepare_data(texts, labels)
dataset = NERDataset(texts, labels, vocab, tag_to_idx)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)



# Initialize model, optimizer, and device
import pickle
model_entity = pickle.load(open("ilgilikisimcikarici2.ai","rb"))



test_sentence = "Vatan Bilgisayardan çok iyi bir Türk Telekomdan daha iyi"
test_tensor = torch.tensor([vocab.get(word, vocab['<UNK>']) for word in test_sentence.split()], dtype=torch.long).to(device)
with torch.no_grad():
    best_tags = model_entity(test_tensor.unsqueeze(0))[0]

idx_to_tag = {i: tag for tag, i in tag_to_idx.items()}
predicted_labels = [idx_to_tag[i] for i in best_tags]
print("Test cümlesi:", test_sentence)
print("Tahmin edilen etiketler:", predicted_labels)

