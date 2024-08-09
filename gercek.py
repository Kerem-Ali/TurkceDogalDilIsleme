import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Önceden eğitilmiş modeli yükleme
with open("yuzde65sentiment.ai", "rb") as f:
    model_sentiment = pickle.load(f)

# Tokenizer ve LabelEncoder'ı yükleme
with open("tokenizer.pickle", "rb") as f:
    tokenizer_sentiment = pickle.load(f)
with open("labelencoder.pickle", "rb") as f:
    le_sentiment = pickle.load(f)

# Örnek kullanım
durumlar = ["negatif","nötr","pozitif"]
# Tahmin yapmak için fonksiyon
def predict_sentiment(text):
    global durumlar
    sequence = tokenizer_sentiment.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=300)
    prediction = model_sentiment.predict(padded)
    return durumlar[le_sentiment.inverse_transform([np.argmax(prediction)])[0]]




from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import torch



from zemberek import (
    TurkishSpellChecker,
    TurkishSentenceNormalizer,
    TurkishMorphology,
    TurkishTokenizer
)

morphology = TurkishMorphology.create_with_defaults()
normalizer = TurkishSentenceNormalizer(morphology)





# 4. Model seçimi ve eğitimi
model_name = "t5-small"  # Daha büyük modeller için "t5-base" veya "t5-large" kullanabilirsiniz
tokenizer_summary = AutoTokenizer.from_pretrained(model_name)

import pickle
model_summary = pickle.load(open("ilgilikisimcikarici.ai","rb"))

def generate_summary(text):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_summary.to(device)
    inputs = tokenizer_summary(text, return_tensors='pt', max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    summary_ids = model_summary.generate(inputs['input_ids'], max_length=64, num_beams=4, early_stopping=True)
    return tokenizer_summary.decode(summary_ids[0], skip_special_tokens=True)
# Test
test_sentence = "berqnet teoride guzel calisiyor fakat gercek dunyada calismasi biraz sikintili ."
test_keyword = "berqnet"
test_input = f"Sentence: {test_sentence} Keyword: {test_keyword}"

generated_summary = generate_summary(test_input)



normalized = normalizer.normalize(generated_summary)
normalized = normalized.replace("."," ")
sc = TurkishSpellChecker(morphology)
"""for word in normalized.split():
    suggesteds = sc.suggest_for_word(word)
    if suggesteds:
        print(suggesteds)
        if word in suggesteds:
            print(word)
        else:
            print(suggesteds[0])"""




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
model_entity = pickle.load(open("entity_cikarici_ellenmemis.ai","rb"))


def extract_entities(sentence, tags):
    entities = []
    current_entity = []
    
    for word, tag in zip(sentence.split(), tags):
        if tag == 'entity':
            current_entity.append(word)
        else:
            if current_entity:
                entities.append(' '.join(current_entity))
                current_entity = []
    
    # To handle the case where the last word(s) in the sentence are part of an entity
    if current_entity:
        entities.append(' '.join(current_entity))
    
    return entities






def extract_entities(sentence, tags):
    entities = []
    current_entity = []
    
    for word, tag in zip(sentence.split(), tags):
        if tag == 'entity':
            current_entity.append(word)
        else:
            if current_entity:
                entities.append(' '.join(current_entity))
                current_entity = []
    
    if current_entity:
        entities.append(' '.join(current_entity))
    
    return entities


def get_entities(text):
    device = torch.device("cuda")
    test_tensor = torch.tensor([vocab.get(word, vocab['<UNK>']) for word in text.split()], dtype=torch.long).to(device)
    with torch.no_grad():
        best_tags = model_entity(test_tensor.unsqueeze(0))[0]
    
    idx_to_tag = {i: tag for tag, i in tag_to_idx.items()}
    predicted_labels = [idx_to_tag[i] for i in best_tags]
    words = text.split(" ")
    return extract_entities(text,predicted_labels)



def remove_stopwords(text):
    import re

    # Türkçe stop words listesi
    turkish_stopwords = ['a', 'acaba', 'açık', 'acıksa', 'açıkça', 'açıkçası', 'adeta', 'af', 'aile', 'aileyle', 'aile ile', 'ainception', 'aira', 'aisa', 'akademik', 'alfa', 'altı', 'altında', 'ama', 'ancak', 'ansızın', 'ant', 'arga', 'artık', 'asla', 'aslında', 'ast', 'ayrı', 'ayrıca', 'az', 'bana', 'baş', 'başka', 'başkası', 'bayağı', 'bazı', 'bazıları', 'be', 'bende', 'beni', 'benim', 'beri', 'beş', 'bile', 'bin', 'bir', 'biraz', 'birbiri', 'birc', 'birc sey', 'biri', 'birkaç', 'birşey', 'birşeyi', 'biz', 'bizden', 'bize', 'bizi', 'bizim', 'bo', 'böyle', 'böylece', 'bu', 'buna', 'bunda', 'bundan', 'bunu', 'bunun', 'burada', 'bütün', 'ca', 'cak', 'çok', 'çünkü', 'da', 'daha', 'dan', 'dcsb', 'de', 'değil', 'değilse', 'demek', 'demiş', 'den', 'değil mi', 'diye', 'diğer', 'diğeri', 'diğerleri', 'diye', 'dolayı', 'dolayısıyla', 'düz', 'eden', 'edip', 'eğer', 'elbette', 'en', 'etmek', 'etrafında', 'ettiği', 'evet', 'f', 'fakat', 'filhal', 'genellikle', 'gibi', 'göre', 'hem', 'henüz', 'hep', 'hepsi', 'hepsini', 'her', 'herhangi', 'herkes', 'herşey', 'hiç', 'hiçbir', 'için', 'iken', 'illa', 'ile', 'ilan', 'ilgili', 'ince', 'içerde', 'içeride', 'içeriye', 'içersinde', 'içerişinde', 'içinde', 'inanılmaz', 'ise', 'işte', 'i̇', 'i̇çin', 'i̇le', 'kadar', 'karşı', 'katip', 'kendi', 'kendilerine', 'kendini', 'kendisi', 'kendisine', 'kendisini', 'her', 'ki', 'kim', 'kime', 'kimin', 'kimisi', 'kısaca', 'köy', 'lerde', 'leri', 'ler', 'madem', 'mi', 'mı', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nedenle', 'nerde', 'nerede', 'nereye', 'nesi', 'nette', 'niçin', 'niye', 'npk', 'o', 'olan', 'olarak', 'oldu', 'olduğu', 'olduğunu', 'olması', 'olur', 'olursa', 'oluyor', 'oluştu', 'on', 'ona', 'ondan', 'onlar', 'onlardan', 'onları', 'onların', 'onu', 'onun', 'orada', 'oysa', 'öbür', 'öc', 'öyle', 'öylece', 'öyleyse', 'önce', 'öteki', 'ötürü', 'pek', 'rağmen', 's', 'sa', 'sadece', 'sanki', 'sanki da', 'sanmak', 'şart', 'şayet', 'şey', 'şeyden', 'şeyler', 'şeyi', 'şöyle', 'tarafından', 'tarzı', 'tek', 'teki', 'tüm', 'tüm ile', 'üzere', 'var', 'vardı', 've', 'veya', 'ya', 'yada', 'yani', 'yapacak', 'yapılan', 'yapılması', 'yapıyor', 'yapıyordu', 'yaptı', 'yaptığı', 'yaptığınız', 'yaptıkları', 'yaptım', 'yaptırdı', 'yaptırdığı', 'yaptırdıkları', 'yapsın', 'yapıyor', 'yaptı', 'yaptılar', 'yaptırdı', 'yaptırmak', 'yarar', 'yardımcı', 'ye', 'yeterki', 'yine', 'yoksa', 'zaten', 'zira']

    def remove_turkish_stopwords(text):
        # Metindeki tüm küçük harflere çevir
        text = text.lower()
        
        # Stopwords'leri sil
        words = [word for word in text.split() if word not in turkish_stopwords]
        
        # Kelimeleri tekrar birleştir
        cleaned_text = ' '.join(words)
        
        # Regex ile tüm noktalama işaretlerini sil
        cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
        
        return cleaned_text

    # Örnek metni temizle
    cleaned_text = remove_turkish_stopwords(text)
    return cleaned_text


def get_sentiments(text):
    words = text.split()
    sentiments = set()
    text = remove_stopwords(text)
    if len(words)%2!=0:
        words.append(" ")
    for i in range(int(len(text)//2),2):
        sentiments.add(predict_sentiment(words[i:i+2]))

    return sentiments



from trnlp import TrnlpWord

def get_lasts_rooted(text):

    kelimeler = text.split()
    
    if len(kelimeler)>0:
        for kelime in kelimeler:
            analiz = TrnlpWord()
            analiz.setword(kelimeler[-1])
            kelimeler[-1] = analiz.get_stem
    
    return " ".join(kelimeler)


yorum = "Türk telekomun allah belasını versin "
entities = get_entities(yorum)


import string

def noktalama_isaretlerini_kaldir(metin):
    translator = str.maketrans('', '', string.punctuation)
    return metin.translate(translator)





import json
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

class Item(BaseModel):
    text: str = Field(..., example="""Fiber 100mb SuperOnline kullanıcısıyım yaklaşık 2 haftadır @Twitch @Kick_Turkey gibi canlı yayın platformlarında 360p yayın izlerken donmalar yaşıyoruz. Başka hiç bir operatörler bu sorunu yaşamazken ben parasını verip alamadığım hizmeti neden ödeyeyim ? @Turkcell """)

@app.post("/predict/", response_model=dict)
async def predict(item: Item):
    yorum = item.text
    yorum = noktalama_isaretlerini_kaldir(yorum)
    entities = get_entities(yorum)
    result = {
        "entity_list": [],
        "results": []
    }
    for entity in entities:
        new_entity = get_lasts_rooted(entity)
        result["entity_list"].append(new_entity)
        test_input = f"Sentence:{yorum} Keyword:{new_entity}"
        ilgili_kisim = generate_summary(test_input)
        ilgililer = []
        for word in ilgili_kisim.split():
            if word.isdigit():
                ilgililer.append(word)
                continue
            suggesteds = sc.suggest_for_word(word)
            if suggesteds:
                if word in suggesteds:
                    ilgililer.append(word)
                else:
                    ilgililer.append(suggesteds[0])
        ilgili_kisim = " ".join(ilgililer)
        #ilgili_kisim = remove_stopwords(ilgili_kisim)
        sentiment = predict_sentiment(ilgili_kisim)
        sentiments = get_sentiments(ilgili_kisim)
        if sentiment not in sentiments:
            result["results"].append({"entity": new_entity, "sentiment": sentiment})
        for sentiment in sentiments:
            result["results"].append({"entity": new_entity, "sentiment": sentiment})

    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



