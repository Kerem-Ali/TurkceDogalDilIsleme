# Türkçe Duygu Analizi ve Varlık Çıkarma Projesi

Bu proje, Türkçe metinler üzerinde duygu analizi ve varlık çıkarma işlemlerini gerçekleştiren bir API sunar.

## Gereksinimler

Projeyi çalıştırmak için aşağıdaki kütüphanelere ihtiyacınız vardır:

- pandas
- numpy
- tensorflow
- scikit-learn
- torch
- transformers
- zemberek-python
- datasets
- torchcrf
- trnlp
- fastapi
- uvicorn

Bu bağımlılıkları yüklemek için:
pip install pandas numpy tensorflow scikit-learn torch transformers zemberek-python datasets torchcrf trnlp fastapi uvicorn

## Kurulum ve Çalıştırma

1. Repo'yu klonlayın:
2. 2. Gerekli modelleri indirin:
- `yuzde65sentiment.ai`
- `tokenizer.pickle`
- `labelencoder.pickle`
- `ilgilikisimcikarici.ai`
- `entity_cikarici_ellenmemis.ai`

Bu dosyaları projenin kök dizinine yerleştirin.

3. API'yi başlatın:

4. API varsayılan olarak `http://0.0.0.0:8000` adresinde çalışacaktır.

## API Kullanımı

API'ye POST isteği göndererek analiz yapabilirsiniz:

Örnek istek gövdesi:
```json
{
  "text": "Fiber 100mb SuperOnline kullanıcısıyım yaklaşık 2 haftadır @Twitch @Kick_Turkey gibi canlı yayın platformlarında 360p yayın izlerken donmalar yaşıyoruz. Başka hiç bir operatörler bu sorunu yaşamazken ben parasını verip alamadığım hizmeti neden ödeyeyim ? @Turkcell"
}
