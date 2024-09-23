import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

# Train verilerini yukleme (örneğin tab ile ayrılmışsa delimiter='\t' kullan)
train_frame = pd.read_csv("FOLIO/data/v0.0/folio-train.txt", delimiter='\t', encoding='latin1')

# Validation verilerini yükleme
validation_frame = pd.read_csv('FOLIO/data/v0.0/folio-validation.txt', delimiter='\t', encoding='latin1')

# Gerekli sütunları seçme
filtered_train = train_frame[['conclusion', 'premises', 'premises-FOL', 'label']]


# print(train_frame.shape) # (1003, 7)
# print(validation_frame.shape) # (204, 5)

print(filtered_train.shape) # (1003, 4)

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Input, Dropout, Concatenate
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertModel
from keras.models import Model

# BERT Tokenizer and Model Initialization
# BertTokenizer: Giriş metinlerini BERT modelinin anlayabileceği formata dönüştürür.
# 				 Metinleri tokenize ederek input_ids ve attention_mask elde eder.
# TFBertModel: BERT modelinin TensorFlow versiyonudur. Giriş olarak metin alır ve
# 			   metnin anlamını yakalayan vektörler (embedding'ler) üretir.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Text Input Processing (for premises and conclusion)
# Bu fonksiyon, BERT tokenizer’ı kullanarak bir metni tokenize eder ve
# geri dönen input_ids ve attention_mask vektörlerini döndürür:
# input_ids: Metindeki her kelimeyi temsil eden numaralardan oluşan bir dizidir.
# attention_mask: Modelin hangi token'lara dikkat etmesi gerektiğini belirleyen bir maske.
def encode_text(text, tokenizer, max_length=128):
    inputs = tokenizer(text, return_tensors="tf", padding='max_length', truncation=True, max_length=max_length)
    return inputs['input_ids'], inputs['attention_mask']

# Example Input
# Input katmanları modelin aldığı girişleri tanımlar. Bu örnekte iki tür metin verisi giriyoruz:
# premise_text: Öncülün metni (128 uzunluğunda bir tamsayı dizisi).
# premise_mask: Öncülün dikkat maskesi.
# conclusion_text: Sonucun metni.
# conclusion_mask: Sonucun dikkat maskesi.
# Bu girişler BERT modeline beslenir, böylece metin verilerinden embedding'ler oluşturulabilir.
premise_text = Input(shape=(128,), dtype=tf.int32, name='premise_text')
premise_mask = Input(shape=(128,), dtype=tf.int32, name='premise_mask')
conclusion_text = Input(shape=(128,), dtype=tf.int32, name='conclusion_text')
conclusion_mask = Input(shape=(128,), dtype=tf.int32, name='conclusion_mask')

# BERT Embedding Layer
# premise_embedding ve conclusion_embedding, BERT modelinden gelen çıkışlardır. BERT modeline metin ve maske verildiğinde,
# bu metnin temsili olarak bir vektör döner.
# [CLS] token embedding: BERT modelinin her metin için en önemli bilgi temsilini yakaladığı [CLS] (classification) token'ı kullanılır.
# Bu token’ın embedding’i, cümlenin genel anlamını yakalar.
premise_embedding = bert_model([premise_text, premise_mask])[0][:, 0, :]  # [CLS] token embedding
conclusion_embedding = bert_model([conclusion_text, conclusion_mask])[0][:, 0, :]

# FOL Input Processing (for premises_FOL and conclusion_FOL)
# premise_fol_input ve conclusion_fol_input, birinci dereceden mantık ifadelerini temsil eden girdilerdir.
# Bu ifadeler de metin girdileri gibi vektörlere dönüştürülecektir.
premise_fol_input = Input(shape=(128,), dtype=tf.int32, name='premise_fol')
conclusion_fol_input = Input(shape=(128,), dtype=tf.int32, name='conclusion_fol')

# FOL Embedding Layer (Simple Embedding for symbolic logic inputs)
# Dense Layer: Bu katman, giriş verisini (FOL verileri) işleyerek 64 boyutlu bir vektöre dönüştürür.
# Activation 'relu': Aktivasyon fonksiyonu, sinir ağında lineer olmayan ilişkiyi sağlayarak daha karmaşık yapıları öğrenmeye olanak tanır.
fol_embedding_layer = Dense(64, activation='relu')
premise_fol_embedding = fol_embedding_layer(premise_fol_input)
conclusion_fol_embedding = fol_embedding_layer(conclusion_fol_input)

# Combine Text and FOL embeddings
# Concatenate: Öncüller (premises) ve sonuçlar (conclusions) için hem metin verilerini hem de
# mantıksal ifadeleri içeren embedding'ler birleştirilir. Böylece, hem doğal dil hem de 
# mantıksal temsillerden elde edilen bilgiler model tarafından kullanılabilir.
combined_premise = Concatenate()([premise_embedding, premise_fol_embedding])
combined_conclusion = Concatenate()([conclusion_embedding, conclusion_fol_embedding])

# Concatenate Premises and Conclusions embeddings
# final_representation: Öncül ve sonuç embedding'leri birleştirilir.
#  Artık model hem öncülleri hem de sonuçları bir arada işleyebilir.
final_representation = Concatenate()([combined_premise, combined_conclusion])

# Classification Layer
# İlk katman modelin öğrenmesi için geniş bir temsil oluşturur.
# Ardından, daha küçük bir boyuta indirgenir. Her iki katmanda da ReLU aktivasyonu kullanılır.
# %20 dropout kullanarak overfitting'i önler. Bu, modelin bazı nöronlarını rastgele kapatarak, modelin genelleme yeteneğini artırır.
# Çıktı katmanında softmax aktivasyonu kullanılarak üç sınıftan birine ait olma olasılığı hesaplanır
x = Dense(256, activation='relu')(final_representation)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
output = Dense(3, activation='softmax', name='output')(x)  # 3 classes: True, False, Uncertain

# Model Definition
model = Model(inputs=[premise_text, premise_mask, conclusion_text, conclusion_mask, premise_fol_input, conclusion_fol_input], outputs=output)

# Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model Summary
model.summary()

