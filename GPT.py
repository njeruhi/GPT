import string

import tensorflow as tf
import numpy as np

import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras

# Hyperparameters: Alter to your suit
batch_size = 16
block_size = 32
use_bias = False
real_data = True
n_embed = 64
num_head = 6
n_layer = 4
learning_rate = 1e-3
max_iters = 1500
eval_iters = 200
drop_out = 0.0
eval_interval = 100
tf.config.run_functions_eagerly(True)
num_epochs = 100

df = pd.read_csv(filepath_or_buffer='conversation.txt', delimiter='/t')
df.head(10)

# This data is in a conversation format.
# So at every other line it starts with a "human_1", "human_2" keywords
# inorder for the model to not keep producing those words we will have to strip the off
df['Human 1: Hi!'] = df['Human 1: Hi!'].str.replace(r'^\w+\s\d+:\s', '', regex=True)

# we return the stripped dataframe to the same textfile path
df.to_csv('conversation.txt', index=False, header=False, sep='\t')

path_to_file = 'conversation.txt'
# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# strip off the punctuation marks to reduce noise
translator = str.maketrans('', '', string.punctuation)
text = text.translate(translator)


text = text
# Define a translation table to remove unwanted characters
translation_table = str.maketrans('', '', '¦©¹¼ÃâæçïðœšŸž˜‚†•‰›€™')

# Apply the translation to remove the characters
text = text.translate(translation_table)

chars = sorted(list(set(text)))
vocab_size = len(chars)

# we vectorize the text data using our data
# we enumerate the chars in the data and give tokens according to their relative 
# position in the data
# e.g letteh h gets an encoding no. of 44 from avocab size of 63
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = tf.constant(encode(text))

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


# this function helps in splitting the tokens by one
# shifted position helping the model not to see future tokens
# because that is what we're predicting 
def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = tf.random.uniform((batch_size,), maxval=len(data) - block_size, dtype=tf.int32)
  x = tf.stack([data[i:i+block_size] for i in ix])
  y = tf.stack([data[i+1:i+block_size+1] for i in ix])

  return x, y


@tf.function
def estimate_loss():
    out = {}
    model.trainable = False
    for split in ['train', 'val']:
        losses = tf.TensorArray(dtype=tf.float32, size=eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses = losses.write(k, loss)
        losses = losses.stack()
        out[split] = tf.reduce_mean(losses)
    model.trainable = True
    return out

class Head(tf.keras.layers.Layer):
    def __init__(self, head_size, n_embd, dropout):
        super(Head, self).__init__()
        self.key = tf.keras.layers.Dense(head_size, use_bias=False)
        self.query = tf.keras.layers.Dense(head_size, use_bias=False)
        self.value = tf.keras.layers.Dense(head_size, use_bias=False)
        self.tril = tf.constant(tf.linalg.band_part(tf.ones((block_size, block_size)), -1, 0))
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # Compute attention scores ('affinities')
        wei = tf.matmul(q, tf.transpose(k, perm=[0, 2, 1])) * tf.math.sqrt(tf.cast(C, dtype=q.dtype))  # B, T, C @ B, C, T --> B, T, T
        wei = tf.where(self.tril[:T, :T] == 0, tf.constant(float('-inf'), dtype=wei.dtype), wei)
        wei = tf.nn.softmax(wei, axis=-1)  # B, T, T
        wei = self.dropout(wei)
        # Perform the aggregation of the values
        v = self.value(x)  # B, T, T @ B, T, C --> B, T, C
        out = tf.matmul(wei, v)
        return out

# Here we are just putting together the attention heads
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, head_size, n_embd, dropout):
        super(MultiHeadAttention, self).__init__()
        self.heads = [Head(head_size, n_embd, dropout) for _ in range(num_heads)]
        self.proj = tf.keras.layers.Dense(n_embd)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x):
        # we concatenate the output vectors (q, k, v) from the attention heads
        out = tf.concat([h(x) for h in self.heads], axis=-1)
        out = self.dropout(self.proj(out))
        return out


# A simple feed forward neural net
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, n_embd, dropout):
        super(FeedForward, self).__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(4 * n_embd, activation='relu'),
            tf.keras.layers.Dense(n_embd),
            tf.keras.layers.Dropout(dropout),
        ])

    def call(self, x):
        return self.net(x)
    
class Block(tf.keras.layers.Layer):
    def __init__(self, n_embd, num_heads, dropout):
        super(Block, self).__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size, n_embd, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class BigramLanguageModel(tf.keras.Model):
    def __init__(self, vocab_size, n_embd, block_size, num_heads, n_layer, dropout):
        super(BigramLanguageModel, self).__init__()
        self.token_embedding_table = tf.keras.layers.Embedding(vocab_size, n_embd)
        self.position_embedding_table = tf.keras.layers.Embedding(block_size, n_embd)
        self.blocks = tf.keras.Sequential([Block(n_embd, num_heads, dropout) for _ in range(n_layer)])
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.lm_head = tf.keras.layers.Dense(vocab_size)

    def call(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(tf.range(T, dtype=tf.float32))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = tf.reshape(logits, (B * T, C))
            targets = tf.reshape(targets, (B * T,))
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))

        return logits, loss


model = BigramLanguageModel(vocab_size, n_embed, block_size, num_head, n_layer, drop_out)

max_new_tokens = 1000

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

for iteration in range(max_iters):

    # Every once in a while, evaluate the loss on the train and val sets
    if iteration % eval_interval == 0 or iteration == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    with tf.GradientTape() as tape:
        logits, loss = model(xb, yb)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Generate from the model
print(model.summary())
context = tf.zeros((1, 1), dtype=tf.int32)
for _ in range(max_new_tokens):
    # crop context to the last block_size tokens
    context_cond = context[:, -block_size:]
    # get the predictions
    logits, _ = model(context_cond)
    # focus only on the last time step
    logits = logits[:, -1, :]
    # apply softmax to get probabilities
    probs = tf.nn.softmax(logits, axis=-1)
    next_token = tf.random.categorical(tf.math.log(probs), 1)
    # cast the tensor to int32
    next_token = tf.cast(next_token, dtype=tf.int32)
    # append sampled index to the running sequence
    context = tf.concat([context, next_token], axis=1)

context_tuple = tuple(context.numpy().flatten())
print(decode(context_tuple))
