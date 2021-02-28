
import tensorflow as tf 
import tensorflow_datasets as tfds 
import numpy as np 
import pickle

# no load data 
imdb , info = tfds.load("imdb_reviews",with_info=True,as_supervised=True)

train_data = imdb["train"]
test_data = imdb["test"]

vocab_size = 10000
oov_tok = "<oov>"
max_length = 120 
embedding_length = 16
trunc_type='post'

training_sentences = []
training_label = []
testing_sentences=[]
testing_label=[]
for s,l in train_data:
  training_sentences.append(s.numpy().decode('utf8'))
  training_label.append(l.numpy())

for s,l in test_data:
  testing_sentences.append(s.numpy().decode('utf8'))
  testing_label.append(l.numpy())

x = np.count_nonzero(training_label)
x

training_label = np.array(training_label)
testing_label = np.array(testing_label)

training_label.size

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(oov_token =oov_tok,num_words=vocab_size)
tokenizer.fit_on_texts(training_sentences)
indices = tokenizer.word_index
training_sequence = tokenizer.texts_to_sequences(training_sentences)
train_padding = pad_sequences(training_sequence,padding='post',maxlen = max_length,truncating=trunc_type)

testing_sequence = tokenizer.texts_to_sequences(testing_sentences)
test_padding = pad_sequences(testing_sequence,padding='post',maxlen=max_length)

model = tf.keras.Sequential([
                             tf.keras.layers.Embedding(vocab_size , embedding_length,input_length=max_length),
                             tf.keras.layers.LSTM(60),
                             tf.keras.layers.Dense(10,activation="relu"),
                             tf.keras.layers.Dense(1,activation="sigmoid")
])

model.summary()

model.compile(optimizer="adam",loss=tf.keras.losses.binary_crossentropy,metrics=['accuracy'])

model.fit(train_padding,training_label,epochs=10,validation_data=(test_padding,testing_label))

b = ["worst"]
a_sequence = tokenizer.texts_to_sequences(b)
a_padding = pad_sequences(a_sequence,maxlen=max_length,padding='post',truncating=trunc_type)
model.predict(a_padding)

model.save('model.h5')
pickle.dump(tokenizer,open('tokenizer.pkl','wb'))
