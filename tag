import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gensim.models import Word2Vec

corpus = ['Farewells are sad',
          'Sadness is tears',
          'Joy is happiness',
          'happiness is pleasure',
          'Hip-hop have swag',
          'refinement is sensational',
          'Beer at the Hanriver',
          'Beer is alcohol',
          'Farewells are tears',
          'Joy is pleasure',
          'Trendy is sensational',
          'refinement is Trendy',
          'Swag is cool',
          'Hip-hop is cool' ]

def remove_stop_words(corpus):
  stop_words = ['are', 'is', 'have', 'is', 'at', 'the']
  result = []

  for text in corpus:
    tmp = text.split(' ')
    for stop_word in stop_words:
      if stop_word in tmp:
        tmp.remove(stop_word)
    result.append(' '.join(tmp))
  return result

corpus = remove_stop_words(corpus)
corpus

words = []
for text in corpus:
  for word in text.split(' '):
    words.append(word)

words = set(words)
words

word2int = {}
for i, word in enumerate(words):
  word2int[word]= i

word2int

sentences = []
for sentence in corpus:
  sentences.append(sentence.split())

WINDOW_SIZE = 3
data = []
for sentence in sentences:
  for idx, word in enumerate(sentence):
    for neighbor in sentence[max(idx-WINDOW_SIZE, 0): min(idx-WINDOW_SIZE, len(sentence))+1]:
      if neighbor != word:
        data.append([word, neighbor])

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)

model.save("tag_w2v.model")

df = pd.DataFrame(data, columns = ['input', 'label'])
df.head(14)

ONE_HOT_DIM = len(words)

def to_one_hot_encoding(data_point_index):
  one_hot_encoding = np.zeros(ONE_HOT_DIM)
  one_hot_encoding[data_point_index] = 1
  return one_hot_encoding

X = []
Y = []

for x, y in zip(df['input'], df['label']):
  X.append(to_one_hot_encoding(word2int[x]))
  Y.append(to_one_hot_encoding(word2int[y]))

X_train = np.array(X)
Y_train = np.array(Y)

encoding_dim = 2
input_word = Input(shape=(ONE_HOT_DIM,))
encoded = Dense(encoding_dim, use_bias=False)(input_word)
decoded = Dense(ONE_HOT_DIM, activation='softmax')(encoded)

w2v_model = Model(input_word, decoded)
w2v_model.compile(optimizer='adam', loss='categorical_crossentropy')

w2v_model.fit(X_train, Y_train, epochs=1000, shuffle=True, verbose=1)

w2v_model.summary()

vectors = w2v_model.layers[1].weights[0].numpy().tolist()
vectors

w2v_df =pd.DataFrame(vectors, columns=['x1', 'x2'])
w2v_df['word'] = list(words)
w2v_df = w2v_df[['word', 'x1', 'x2']]
w2v_df

fig, ax = plt.subplots()

for word, x1, x2 in zip(w2v_df['word'], w2v_df['x1'], w2v_df['x2']):
  ax.annotate(word, (x1,x2))

PADDING = 1.0
x_axis_min = np.min(vectors, axis=0)[0] - PADDING
y_axis_min = np.min(vectors, axis=0)[1] - PADDING
x_axis_max = np.max(vectors, axis=0)[0] + PADDING
y_axis_max = np.max(vectors, axis=0)[1] + PADDING

plt.xlim(x_axis_min, x_axis_max)
plt.ylim(y_axis_min, y_axis_max)
plt.rcParams["figure.figsize"] = (9,9)

plt.show()

model = Word2Vec.load('tag_w2v')

#가장 유사한 단어 추출
print(model.wv.most_similar("Hip-hop",topn=3))
