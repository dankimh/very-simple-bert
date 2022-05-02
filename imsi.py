

!pip install transformers

import transformers

import pandas as pd
import numpy as np
import urllib.request
import os
from tqdm import tqdm
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

urllib.request.urlretrieve("https://raw.githubusercontent.com/kocohub/korean-hate-speech/master/labeled/dev.tsv", filename="dev.tsv")
urllib.request.urlretrieve("https://raw.githubusercontent.com/kocohub/korean-hate-speech/master/labeled/train.tsv", filename="train.tsv")

train_data = pd.read_table('train.tsv')
test_data = pd.read_table('dev.tsv')

print('훈련용 리뷰 개수 :',len(train_data))

print('테스트용 리뷰 개수 :',len(test_data))

for i in range(len(train_data)):
  if train_data['hate'][i] == 'none':
    train_data['hate'][i]=1
  else:
    train_data['hate'][i]=0

for i in range(len(test_data)):
  if test_data['hate'][i] == 'none':
    test_data['hate'][i]=1
  else:
    test_data['hate'][i]=0

print(train_data[:5])

print(test_data[:5])

train_data = train_data.dropna(how = 'any') 
train_data = train_data.reset_index(drop=True)
print(train_data.isnull().values.any())

test_data = test_data.dropna(how = 'any') 
test_data = test_data.reset_index(drop=True)
print(test_data.isnull().values.any())

print(len(train_data))

print(len(test_data))

tokenizer = BertTokenizer.from_pretrained('klue/bert-base')

max_seq_len = 128

def convert_examples_to_features(examples, labels, max_seq_len, tokenizer):
    
    input_ids, attention_masks, token_type_ids, data_labels = [], [], [], []
    
    for example, label in tqdm(zip(examples, labels), total=len(examples)):
        input_id = tokenizer.encode(example, max_length=max_seq_len, pad_to_max_length=True)
        padding_count = input_id.count(tokenizer.pad_token_id)
        attention_mask = [1] * (max_seq_len - padding_count) + [0] * padding_count
        token_type_id = [0] * max_seq_len

        assert len(input_id) == max_seq_len, "Error with input length {} vs {}".format(len(input_id), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_id) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_id), max_seq_len)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        data_labels.append(label)

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)

    data_labels = np.asarray(data_labels, dtype=np.int32)

    return (input_ids, attention_masks, token_type_ids), data_labels


train_X, train_y = convert_examples_to_features(train_data['comments'], train_data['hate'], max_seq_len=max_seq_len, tokenizer=tokenizer)

test_X, test_y = convert_examples_to_features(test_data['comments'], test_data['hate'], max_seq_len=max_seq_len, tokenizer=tokenizer)

# 최대 길이: 128
input_id = train_X[0][0]
attention_mask = train_X[1][0]
token_type_id = train_X[2][0]
label = train_y[0]

model = TFBertModel.from_pretrained("klue/bert-base", from_pt=True)

max_seq_len = 128

input_ids_layer = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32)
attention_masks_layer = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32)
token_type_ids_layer = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32)

outputs = model([input_ids_layer, attention_masks_layer, token_type_ids_layer])

print(outputs)

print(outputs[0])

print(outputs[1])

class TFBertForSequenceClassification(tf.keras.Model):
    def __init__(self, model_name):
        super(TFBertForSequenceClassification, self).__init__()
        self.bert = TFBertModel.from_pretrained(model_name, from_pt=True)
        self.classifier = tf.keras.layers.Dense(1,
                                                kernel_initializer=tf.keras.initializers.TruncatedNormal(0.02),
                                                activation='sigmoid',
                                                name='classifier')

    def call(self, inputs):
        input_ids, attention_mask, token_type_ids = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_token = outputs[1]
        prediction = self.classifier(cls_token)

        return prediction


resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)

strategy = tf.distribute.experimental.TPUStrategy(resolver)

with strategy.scope():
  model = TFBertForSequenceClassification("klue/bert-base")
  optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
  loss = tf.keras.losses.BinaryCrossentropy()
  model.compile(optimizer=optimizer, loss=loss, metrics = ['accuracy'])

model.fit(train_X, train_y, epochs=2, batch_size=64, validation_split=0.2)

results = model.evaluate(test_X, test_y, batch_size=1024)
print("test loss, test acc: ", results)
