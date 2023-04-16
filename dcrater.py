
import re

a = re.sub(r'^\d+\s*', '', a, flags=re.MULTILINE)

# Print the updated string
print(a)

# upload data files to Colab
from google.colab import files
uploaded = files.upload()
# verify files present
# !ls -l
# # install needed packages
# # more on tensorflow_text at https://www.tensorflow.org/text
# !pip install -q -U tensorflow-text
# !pip install -q tensorflow_datasets
# !pip install googletrans==4.0.0-rc1
# import needed libraries
import collections
import logging
import os
import pathlib
import re
import string
import sys
import tempfile
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction30 from googletrans import Translator
# build TF datasets from input sentences in both languages
lines_dataset_en = tf.data.TextLineDataset("new.en")
lines_dataset_he = tf.data.TextLineDataset("new.he")
NUM_PAIRS = 0
for en in lines_dataset_en:
    NUM_PAIRS += 1
print('There are ' + str(NUM_PAIRS) + ' samples in training data')
# verify Hebrew file interpreted correctly
for he in lines_dataset_he.take(20):
    print("Hebrew: ", he.numpy().decode('utf-8'))
# verify English file interpreted correctly
for en in lines_dataset_en.take(20):
    print("English: ", en.numpy().decode('utf-8'))
# combine languages into single dataset
combined = tf.data.Dataset.zip((lines_dataset_en, lines_dataset_he))
# verify combined dataset is correct
for en, he in combined.take(20):
    print("English: ", en.numpy().decode('utf-8'))
    print("Hebrew: ", he.numpy().decode('utf-8'))
# disable warnings
tf.get_logger().setLevel('ERROR')
pwd = pathlib.Path.cwd()
# import BERT tool for building tokenizer
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
# set tokenizer parameters and add reserved tokens; input files already lower-cased, but 
#lower_case option does NFD normalization, which is needed
bert_tokenizer_params=dict(lower_case=True)
reserved_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]
# main parameter here that could be tuned is vocab size
bert_vocab_args = dict(
# The target vocabulary size
vocab_size = 16000,
# Reserved tokens that must be included in the vocabulary
reserved_tokens=reserved_tokens,
# Arguments for `text.BertTokenizer`
bert_tokenizer_params=bert_tokenizer_params,
# Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
learn_params={},
)
# build English vocab file (takes several mins)
# this is the bert_vocab module building its vocab file from the raw English sentences
#%%time
en_vocab = bert_vocab.bert_vocab_from_dataset(
lines_dataset_en.batch(1000).prefetch(2),**bert_vocab_args )
# confirm sub-word vocab is built correctly (last line will look strange; this is expected with 
#bert_vocab)
print(en_vocab[:10])
print(en_vocab[100:110])
print(en_vocab[1000:1010])
print(en_vocab[-10:])
# write English vocab to file
# this file will be used to build tokenizer
def write_vocab_file(filepath, vocab):
    with open(filepath, 'w') as f:
        for token in vocab:
            print(token, file=f)
write_vocab_file('en_vocab.txt', en_vocab)
# build Hebrew vocab file (takes several mins)
#%%time
he_vocab = bert_vocab.bert_vocab_from_dataset(
lines_dataset_he.batch(1000).prefetch(2),
**bert_vocab_args
)
# confirm Hebrew sub-word vocab built correctly (last line will look strange; this is expected with
#bert_vocab)
print(he_vocab[:10])
print(he_vocab[100:110])
print(he_vocab[1000:1010])
print(he_vocab[-10:])
# write Hebrew vocab to file and confirm both vocab files now present
# this file will be used to build tokenizer
write_vocab_file('he_vocab.txt', he_vocab)
#!ls
# build both BERT tokenizers
he_tokenizer = text.BertTokenizer('he_vocab.txt', **bert_tokenizer_params)
en_tokenizer = text.BertTokenizer('en_vocab.txt', **bert_tokenizer_params)
# take 3 example sentences to confirm tokenizer working correctly; first print sentences
for en_examples, he_examples in combined.batch(3).take(1):
    for ex in en_examples:
        print(ex.numpy())
    # Now verify tokenization
    # Tokenize the examples -> (batch, word, word-piece)
    en_token_batch = en_tokenizer.tokenize(en_examples)
    # BERT tokenizer returns ragged tensor with dims we don't need
    # Merge the word and word-piece axes of ragged tensor -> (batch, tokens)
    en_token_batch = en_token_batch.merge_dims(-2,-1)
    # confirm tokens match input sentence
    for ex in en_token_batch.to_list():
        print(ex)
    # Verify we can reassemble tokens to recover original sentences using tf.gather()
    # Lookup each token id in the vocabulary.
    txt_tokens = tf.gather(en_vocab, en_token_batch)
    # Join with spaces.
    np.char.decode(tf.strings.reduce_join(txt_tokens, separator=' ', axis=-1).numpy().astype(np.bytes_), 'UTF-8')
    # Now remove token markers
    words = en_tokenizer.detokenize(en_token_batch)
    tf.strings.reduce_join(words, separator=' ', axis=-1)
# Repeat above steps for Hebrew; first print 3 example sentences
for en_examples, he_examples in combined.batch(3).take(1):
    for ex in he_examples:
        print(ex.numpy().decode('utf-8'))
    # Tokenize the examples -> (batch, word, word-piece)
    he_token_batch = he_tokenizer.tokenize(he_examples)
    # Merge the word and word-piece axes -> (batch, tokens)
    he_token_batch = he_token_batch.merge_dims(-2,-1)
    # confirm tokens match input sentences
    for ex in he_token_batch.to_list():
        print(ex)
    # Lookup each token id in the vocabulary.
    txt_tokens = tf.gather(he_vocab, he_token_batch)
    # Join with spaces.
    np.char.decode(tf.strings.reduce_join(txt_tokens, separator=' ', axis=-1).numpy().astype(np.bytes_), 'UTF-8')
    # Remove token markers
    words = he_tokenizer.detokenize(he_token_batch)
    np.char.decode(tf.strings.reduce_join(words, separator=' ', axis=-1).numpy().astype(np.bytes_),'UTF-8')

# Code to add start and end tokens and return ragged tensor
START = tf.argmax(tf.constant(reserved_tokens) == "[START]")
END = tf.argmax(tf.constant(reserved_tokens) == "[END]")

def add_start_end(ragged):
    count = ragged.bounding_shape()[0]
    starts = tf.fill([count,1], START)
    ends = tf.fill([count,1], END)
    return tf.concat([starts, ragged, ends], axis=1)
# Test detokenization with start/end tokens this performs the above steps in two commands
words = en_tokenizer.detokenize(add_start_end(en_token_batch))
tf.strings.reduce_join(words, separator=' ', axis=-1)
# Function to remove reserved tokens after detokenization
def cleanup_text(reserved_tokens, token_txt):
    # Drop the reserved tokens, except for "[UNK]".
    bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]198  bad_token_re = "|".join(bad_tokens)
    bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
    result = tf.ragged.boolean_mask(token_txt, ~bad_cells)
    # Join them into strings.
    result = tf.strings.reduce_join(result, separator=' ', axis=-1)
    return result
# Test round-trip tokenization/detokenization on English
en_token_batch = en_tokenizer.tokenize(en_examples).merge_dims(-2,-1)
words = en_tokenizer.detokenize(en_token_batch)
cleanup_text(reserved_tokens, words).numpy()
# Test round-trip tokenization/detokenization on Hebrew
he_token_batch = he_tokenizer.tokenize(he_examples).merge_dims(-2,-1)
words = he_tokenizer.detokenize(he_token_batch)
np.char.decode(cleanup_text(reserved_tokens, words).numpy().astype(np.bytes_), 'UTF-8')
# Above code establishes the full tokenization/detokenization process
# Now we move to building a language-agnostic structure to tokenize for the Transformer
# Build full language-agnostic tokenizer class; this is standard Google code
class CustomTokenizer(tf.Module):
def __init__(self, reserved_tokens, vocab_path):
self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)
self._reserved_tokens = reserved_tokens
self._vocab_path = tf.saved_model.Asset(vocab_path)
vocab = pathlib.Path(vocab_path).read_text().splitlines()
self.vocab = tf.Variable(vocab)
## Create the signatures for export: 
# Include a tokenize signature for a batch of strings. 
self.tokenize.get_concrete_function(
tf.TensorSpec(shape=[None], dtype=tf.string))
# Include `detokenize` and `lookup` signatures for:
# * `Tensors` with shapes [tokens] and [batch, tokens]
# * `RaggedTensors` with shape [batch, tokens]
self.detokenize.get_concrete_function(
tf.TensorSpec(shape=[None, None], dtype=tf.int64))
self.detokenize.get_concrete_function(
tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))
self.lookup.get_concrete_function(
tf.TensorSpec(shape=[None, None], dtype=tf.int64))
self.lookup.get_concrete_function(
tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))
# These `get_*` methods take no arguments
self.get_vocab_size.get_concrete_function()
self.get_vocab_path.get_concrete_function()
self.get_reserved_tokens.get_concrete_function()254
@tf.function
def tokenize(self, strings):
enc = self.tokenizer.tokenize(strings)
# Merge the `word` and `word-piece` axes.
enc = enc.merge_dims(-2,-1)
enc = add_start_end(enc)
return enc
@tf.function
def detokenize(self, tokenized):
words = self.tokenizer.detokenize(tokenized)
return cleanup_text(self._reserved_tokens, words)
@tf.function
def lookup(self, token_ids):
return tf.gather(self.vocab, token_ids)
@tf.function
def get_vocab_size(self):
return tf.shape(self.vocab)[0]
@tf.function
def get_vocab_path(self):
return self._vocab_path
@tf.function
def get_reserved_tokens(self):
return tf.constant(self._reserved_tokens)
#Instantiate tokenizer class for both Hebrew and English
tokenizers = tf.Module()
tokenizers.en = CustomTokenizer(reserved_tokens, 'en_vocab.txt')
tokenizers.he = CustomTokenizer(reserved_tokens, 'he_vocab.txt')
# Save tokenizer model
model_name = 'crater_translate_en_he_converter'
tf.saved_model.save(tokenizers, model_name)
# Verify tokenizer model can be reloaded
tokenizers = tf.saved_model.load(model_name)
tokenizers.en.get_vocab_size().numpy()
# Verify tokenizer works on test sentence
tokens = tokenizers.en.tokenize(['Hello TensorFlow!'])
text_tokens = tokenizers.en.lookup(tokens)
text_tokens
# Remove token markers to get original sentence
round_trip = tokenizers.en.detokenize(tokens)
print(round_trip.numpy()[0].decode('utf-8'))
# Routine to tokenize sentence pairs; needed to make sentence batches for training
def tokenize_pairs(en, he):
he = tokenizers.he.tokenize(he)
# Convert from ragged to dense, padding with zeros.310  he = he.to_tensor()
en = tokenizers.en.tokenize(en)
# Convert from ragged to dense, padding with zeros.
en = en.to_tensor()
return en, he
# Make training batches
BUFFER_SIZE = 20000
BATCH_SIZE = 300
def make_batches(ds):
return (
ds
.cache()
.shuffle(BUFFER_SIZE)
.batch(BATCH_SIZE)
.map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
.prefetch(tf.data.AUTOTUNE))
train_batches = make_batches(combined)
# Calculate positional encoding (needed for Transformer model to indicate word spacing
#information)
def get_angles(pos, i, d_model):
angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
return pos * angle_rates
# Add positional encoding to data
def positional_encoding(position, d_model):
angle_rads = get_angles(np.arange(position)[:, np.newaxis],
np.arange(d_model)[np.newaxis, :],
d_model)
# apply sin to even indices in the array; 2i
angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
# apply cos to odd indices in the array; 2i+1
angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
pos_encoding = angle_rads[np.newaxis, ...]
return tf.cast(pos_encoding, dtype=tf.float32)
# Visualize positional encoding
n, d = 2048, 512
pos_encoding = positional_encoding(n, d)
print(pos_encoding.shape)
pos_encoding = pos_encoding[0]
# Juggle the dimensions for the plot
pos_encoding = tf.reshape(pos_encoding, (n, d//2, 2))
pos_encoding = tf.transpose(pos_encoding, (2, 1, 0))
pos_encoding = tf.reshape(pos_encoding, (d, n))
plt.pcolormesh(pos_encoding, cmap='RdBu')
plt.ylabel('Depth')366 plt.xlabel('Position')
plt.colorbar()
plt.show()
# Create mask for padding tokens so translator ignores them
def create_padding_mask(seq):
seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
# add extra dimensions to add the padding
# to the attention logits.
return seq[:, tf.newaxis, tf.newaxis, :] # (batch_size, 1, 1, seq_len)
# Verify padding mask works correctly; places 1's where pad tokens (indicated by zeros in input)
#exist
x= tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
create_padding_mask(x)
# Mask future tokens so translator cannot see future
def create_look_ahead_mask(size):
mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
return mask # (seq_len, seq_len)
# Verify look-ahead mask works correctly
x= tf.random.uniform((1, 3))
temp = create_look_ahead_mask(x.shape[1])
temp
# Now start to build Transformer; begin with scaled dot-product attention mechanism, which is 
#core of Transformer
def scaled_dot_product_attention(q, k, v, mask):
Calculate the attention weights.
q, k, v must have matching leading dimensions.
k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
The mask has different shapes depending on its type(padding or look ahead)
but it must be broadcastable for addition.
Args:
q: query shape == (..., seq_len_q, depth)
k: key shape == (..., seq_len_k, depth)
v: value shape == (..., seq_len_v, depth_v)
mask: Float tensor with shape broadcastable
to (..., seq_len_q, seq_len_k). Defaults to None.
Returns:
output, attention_weights
matmul_qk = tf.matmul(q, k, transpose_b=True) # (..., seq_len_q, seq_len_k)
# scale matmul_qk
dk = tf.cast(tf.shape(k)[-1], tf.float32)
scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
# add the mask to the scaled tensor.
if mask is not None:
scaled_attention_logits += (mask * -1e9)422
# softmax is normalized on the last axis (seq_len_k) so that the scores
# add up to 1.
attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # (..., seq_len_q, seq_len_k)
output = tf.matmul(attention_weights, v) # (..., seq_len_q, depth_v)
return output, attention_weights
# Utility print function
def print_out(q, k, v):
temp_out, temp_attn = scaled_dot_product_attention(
q, k, v, None)
print('Attention weights are:')
print(temp_attn)
print('Output is:')
print(temp_out)
# Verify print function
np.set_printoptions(suppress=True)
temp_k = tf.constant([[10, 0, 0],
[0, 10, 0],
[0, 0, 10],
[0, 0, 10]], dtype=tf.float32) # (4, 3)
temp_v = tf.constant([[1, 0],
[10, 0],
[100, 5],
[1000, 6]], dtype=tf.float32) # (4, 2)
# This `query` aligns with the second `key`,
# so the second `value` is returned.
temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32) # (1, 3)
print_out(temp_q, temp_k, temp_v)
# This query aligns with a repeated key (third and fourth),
# so all associated values get averaged.
temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32) # (1, 3)
print_out(temp_q, temp_k, temp_v)
# This query aligns equally with the first and second key,
# so their values get averaged.
temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32) # (1, 3)
print_out(temp_q, temp_k, temp_v)
# One more test case
temp_q = tf.constant([[0, 0, 10],
[0, 10, 0],
[10, 10, 0]], dtype=tf.float32) # (3, 3)
print_out(temp_q, temp_k, temp_v)
# Add attention layers to create multi-head attention
class MultiHeadAttention(tf.keras.layers.Layer):
def __init__(self, d_model, num_heads):
super(MultiHeadAttention, self).__init__()478  self.num_heads = num_heads
self.d_model = d_model
assert d_model % self.num_heads == 0
self.depth = d_model // self.num_heads
self.wq = tf.keras.layers.Dense(d_model)
self.wk = tf.keras.layers.Dense(d_model)
self.wv = tf.keras.layers.Dense(d_model)
self.dense = tf.keras.layers.Dense(d_model)
def split_heads(self, x, batch_size):
Split the last dimension into (num_heads, depth).
Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
x= tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
return tf.transpose(x, perm=[0, 2, 1, 3])
def call(self, v, k, q, mask):
batch_size = tf.shape(q)[0]
q= self.wq(q) # (batch_size, seq_len, d_model)
k= self.wk(k) # (batch_size, seq_len, d_model)
v= self.wv(v) # (batch_size, seq_len, d_model)
q= self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
k= self.split_heads(k, batch_size) # (batch_size, num_heads, seq_len_k, depth)
v= self.split_heads(v, batch_size) # (batch_size, num_heads, seq_len_v, depth)
# scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
# attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
scaled_attention, attention_weights = scaled_dot_product_attention(
q, k, v, mask)
scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3]) # (batch_size, seq_len_q,
num_heads, depth)
concat_attention = tf.reshape(scaled_attention,
(batch_size, -1, self.d_model)) # (batch_size, seq_len_q, d_model)
output = self.dense(concat_attention) # (batch_size, seq_len_q, d_model)
return output, attention_weights
# Verify multi-head attention works
temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
y= tf.random.uniform((1, 60, 512)) # (batch_size, encoder_sequence, d_model)
out, attn = temp_mha(y, k=y, q=y, mask=None)
out.shape, attn.shape
# Add feed-forward layer
def point_wise_feed_forward_network(d_model, dff):
return tf.keras.Sequential([
tf.keras.layers.Dense(dff, activation='relu'), # (batch_size, seq_len, dff)534  tf.keras.layers.Dense(d_model) # (batch_size, seq_len, d_model)
])
# Verify feed-forward layer
sample_ffn = point_wise_feed_forward_network(512, 2048)
sample_ffn(tf.random.uniform((64, 50, 512))).shape
# Build Transformer encoder layer
class EncoderLayer(tf.keras.layers.Layer):
def __init__(self, d_model, num_heads, dff, rate=0.1):
super(EncoderLayer, self).__init__()
self.mha = MultiHeadAttention(d_model, num_heads)
self.ffn = point_wise_feed_forward_network(d_model, dff)
self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
self.dropout1 = tf.keras.layers.Dropout(rate)
self.dropout2 = tf.keras.layers.Dropout(rate)
def call(self, x, training, mask):
attn_output, _= self.mha(x, x, x, mask) # (batch_size, input_seq_len, d_model)
attn_output = self.dropout1(attn_output, training=training)
out1 = self.layernorm1(x + attn_output) # (batch_size, input_seq_len, d_model)
ffn_output = self.ffn(out1) # (batch_size, input_seq_len, d_model)
ffn_output = self.dropout2(ffn_output, training=training)
out2 = self.layernorm2(out1 + ffn_output) # (batch_size, input_seq_len, d_model)
return out2
# Verify encoder layer
sample_encoder_layer = EncoderLayer(512, 8, 2048)
sample_encoder_layer_output = sample_encoder_layer(
tf.random.uniform((64, 43, 512)), False, None)
sample_encoder_layer_output.shape # (batch_size, input_seq_len, d_model)
# Build Transformer decoder layer
class DecoderLayer(tf.keras.layers.Layer):
def __init__(self, d_model, num_heads, dff, rate=0.1):
super(DecoderLayer, self).__init__()
self.mha1 = MultiHeadAttention(d_model, num_heads)
self.mha2 = MultiHeadAttention(d_model, num_heads)
self.ffn = point_wise_feed_forward_network(d_model, dff)
self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
self.dropout1 = tf.keras.layers.Dropout(rate)590  self.dropout2 = tf.keras.layers.Dropout(rate)
self.dropout3 = tf.keras.layers.Dropout(rate)
def call(self, x, enc_output, training,
look_ahead_mask, padding_mask):
# enc_output.shape == (batch_size, input_seq_len, d_model)
attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask) # (batch_size, 
target_seq_len, d_model)
attn1 = self.dropout1(attn1, training=training)
out1 = self.layernorm1(attn1 + x)
attn2, attn_weights_block2 = self.mha2(
enc_output, enc_output, out1, padding_mask) # (batch_size, target_seq_len, d_model)
attn2 = self.dropout2(attn2, training=training)
out2 = self.layernorm2(attn2 + out1) # (batch_size, target_seq_len, d_model)
ffn_output = self.ffn(out2) # (batch_size, target_seq_len, d_model)
ffn_output = self.dropout3(ffn_output, training=training)
out3 = self.layernorm3(ffn_output + out2) # (batch_size, target_seq_len, d_model)
return out3, attn_weights_block1, attn_weights_block2
# Verify decoder layer
sample_decoder_layer = DecoderLayer(512, 8, 2048)
sample_decoder_layer_output, _, _= sample_decoder_layer(
tf.random.uniform((64, 50, 512)), sample_encoder_layer_output,
False, None, None)
sample_decoder_layer_output.shape # (batch_size, target_seq_len, d_model)
# Build Transformer encoder from encoder layers
class Encoder(tf.keras.layers.Layer):
def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
maximum_position_encoding, rate=0.1):
super(Encoder, self).__init__()
self.d_model = d_model
self.num_layers = num_layers
self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
self.pos_encoding = positional_encoding(maximum_position_encoding,
self.d_model)
self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
for _ in range(num_layers)]
self.dropout = tf.keras.layers.Dropout(rate)
def call(self, x, training, mask):
seq_len = tf.shape(x)[1]
# adding embedding and position encoding.
x= self.embedding(x) # (batch_size, input_seq_len, d_model)646  x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
x += self.pos_encoding[:, :seq_len, :]
x= self.dropout(x, training=training)
for i in range(self.num_layers):
x= self.enc_layers[i](x, training, mask)
return x # (batch_size, input_seq_len, d_model)
# Verify Transformer encoder
sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
dff=2048, input_vocab_size=8500,
maximum_position_encoding=10000)
temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)
sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)
print(sample_encoder_output.shape) # (batch_size, input_seq_len, d_model)
# Build Transformer decoder from decoder layers
class Decoder(tf.keras.layers.Layer):
def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
maximum_position_encoding, rate=0.1):
super(Decoder, self).__init__()
self.d_model = d_model
self.num_layers = num_layers
self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
for _ in range(num_layers)]
self.dropout = tf.keras.layers.Dropout(rate)
def call(self, x, enc_output, training,
look_ahead_mask, padding_mask):
seq_len = tf.shape(x)[1]
attention_weights = {}
x= self.embedding(x) # (batch_size, target_seq_len, d_model)
x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
x += self.pos_encoding[:, :seq_len, :]
x= self.dropout(x, training=training)
for i in range(self.num_layers):
x, block1, block2 = self.dec_layers[i](x, enc_output, training,
look_ahead_mask, padding_mask)
attention_weights[f'decoder_layer{i+1}_block1'] = block1
attention_weights[f'decoder_layer{i+1}_block2'] = block2
# x.shape == (batch_size, target_seq_len, d_model)702  return x, attention_weights
# Verify Transformer decoder
sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
dff=2048, target_vocab_size=8000,
maximum_position_encoding=5000)
temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)
output, attn = sample_decoder(temp_input,
enc_output=sample_encoder_output,
training=False,
look_ahead_mask=None,
padding_mask=None)
output.shape, attn['decoder_layer2_block2'].shape
# Build full Transformer
class Transformer(tf.keras.Model):
def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
target_vocab_size, pe_input, pe_target, rate=0.1):
super(Transformer, self).__init__()
self.tokenizer = Encoder(num_layers, d_model, num_heads, dff,
input_vocab_size, pe_input, rate)
self.decoder = Decoder(num_layers, d_model, num_heads, dff,
target_vocab_size, pe_target, rate)
self.final_layer = tf.keras.layers.Dense(target_vocab_size)
def call(self, inp, tar, training, enc_padding_mask,
look_ahead_mask, dec_padding_mask):
enc_output = self.tokenizer(inp, training, enc_padding_mask) # (batch_size, inp_seq_len,
d_model)
# dec_output.shape == (batch_size, tar_seq_len, d_model)
dec_output, attention_weights = self.decoder(
tar, enc_output, training, look_ahead_mask, dec_padding_mask)
final_output = self.final_layer(dec_output) # (batch_size, tar_seq_len, target_vocab_size)
return final_output, attention_weights
#Instantiate transformer with default hyperparameters
sample_transformer = Transformer(
num_layers=2, d_model=512, num_heads=8, dff=2048,
input_vocab_size=8500, target_vocab_size=8000,
pe_input=10000, pe_target=6000)
temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)
fn_out, _= sample_transformer(temp_input, temp_target, training=False,
enc_padding_mask=None,
look_ahead_mask=None,758  dec_padding_mask=None)
fn_out.shape # (batch_size, tar_seq_len, target_vocab_size)
# Set hyperparameters for small Transformer
num_layers = 8
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1
# Set learning rate schedule
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
def __init__(self, d_model, warmup_steps=4000):
super(CustomSchedule, self).__init__()
self.d_model = d_model
self.d_model = tf.cast(self.d_model, tf.float32)
self.warmup_steps = warmup_steps
def __call__(self, step):
arg1 = tf.math.rsqrt(step)
arg2 = step * (self.warmup_steps ** -1.5)
return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
# Instantiate learning rate and set optimizer
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
epsilon=1e-9)
# Visualize learning rate schedule
temp_learning_rate_schedule = CustomSchedule(d_model)
plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
# Define loss and accuracy functions
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
from_logits=True, reduction='none')
def loss_function(real, pred):
mask = tf.math.logical_not(tf.math.equal(real, 0))
loss_= loss_object(real, pred)
mask = tf.cast(mask, dtype=loss_.dtype)
loss_ *= mask
return tf.reduce_sum(loss_)/tf.reduce_sum(mask)
def accuracy_function(real, pred):
accuracies = tf.equal(real, tf.argmax(pred, axis=2))814
mask = tf.math.logical_not(tf.math.equal(real, 0))
accuracies = tf.math.logical_and(mask, accuracies)
accuracies = tf.cast(accuracies, dtype=tf.float32)
mask = tf.cast(mask, dtype=tf.float32)
return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)
# Instantiate small Transformer
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
transformer = Transformer(
num_layers=num_layers,
d_model=d_model,
num_heads=num_heads,
dff=dff,
input_vocab_size=tokenizers.he.get_vocab_size(),
target_vocab_size=tokenizers.en.get_vocab_size(),
pe_input=1000,
pe_target=1000,
rate=dropout_rate)
# Create final masks
def create_masks(inp, tar):
# Encoder padding mask
enc_padding_mask = create_padding_mask(inp)
# Used in the 2nd attention block in the decoder.
# This padding mask is used to mask the encoder outputs.
dec_padding_mask = create_padding_mask(inp)
# Used in the 1st attention block in the decoder.
# It is used to pad and mask future tokens in the input received by
# the decoder.
look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
dec_target_padding_mask = create_padding_mask(tar)
combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
return enc_padding_mask, combined_mask, dec_padding_mask
# Set up training checkpoints
checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(transformer=transformer,
optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
ckpt.restore(ckpt_manager.latest_checkpoint)
print('Latest checkpoint restored!!')
# Choose number of training epochs
EPOCHS = 3870
train_step_signature = [
tf.TensorSpec(shape=(None, None), dtype=tf.int64),
tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]
# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.
@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
tar_inp = tar[:, :-1]
tar_real = tar[:, 1:]
enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
with tf.GradientTape() as tape:
predictions, _= transformer(inp, tar_inp,
True,
enc_padding_mask,
combined_mask,
dec_padding_mask)
loss = loss_function(tar_real, predictions)
gradients = tape.gradient(loss, transformer.trainable_variables)
optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
train_loss(loss)
train_accuracy(accuracy_function(tar_real, predictions))
# Run training! Each epoch takes several mins with GPU
for epoch in range(EPOCHS):
start = time.time()
last_start = start
train_loss.reset_states()
train_accuracy.reset_states()
# inp -> Hebrew, tar -> English
for (batch, (tar, inp)) in enumerate(train_batches):
train_step(inp, tar)
if batch % 50 == 0:
print(f'Epoch {epoch + 1} Batch {batch}/{(NUM_PAIRS / BATCH_SIZE):.0f} Loss 
{train_loss.result():.4f} Accuracy {train_accuracy.result():.4f} ({time.time() - last_start:.1f} secs)')
last_start = time.time()
if (epoch + 1) % 5 == 0:
ckpt_save_path = ckpt_manager.save()
print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')
print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs
')
# Set up evaluation function to translate a new sentence
def evaluate(sentence, max_length=40):
# inp sentence is Hebrew, hence adding the start and end token
sentence = tf.convert_to_tensor([sentence])
sentence = tokenizers.he.tokenize(sentence).to_tensor()
encoder_input = sentence
# as the target is english, the first word to the transformer should be the
# english start token.
start, end = tokenizers.en.tokenize([''])[0]
output = tf.convert_to_tensor([start])
output = tf.expand_dims(output, 0)
for i in range(max_length):
enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
encoder_input, output)
# predictions.shape == (batch_size, seq_len, vocab_size)
predictions, attention_weights = transformer(encoder_input,
output,
False,
enc_padding_mask,
combined_mask,
dec_padding_mask)
# select the last word from the seq_len dimension
predictions = predictions[:, -1:, :] # (batch_size, 1, vocab_size)
predicted_id = tf.argmax(predictions, axis=-1)
# concatentate the predicted_id to the output which is given to the decoder
# as its input.
output = tf.concat([output, predicted_id], axis=-1)
# return the result if the predicted_id is equal to the end token
if predicted_id == end:
break
# output.shape (1, tokens)
text = tokenizers.en.detokenize(output)[0] # shape: ()
tokens = tokenizers.en.lookup(output)[0]
return text, tokens, attention_weights
# Print translation with ground truth
def print_translation(sentence, tokens, ground_truth):
print(f'{"Input:":15s}: {sentence}')
print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
print(f'{"Ground truth":15s}: {ground_truth}')
import re
import nltk982 import nltk.translate.meteor_score as ms
nltk.download('wordnet')
# First test
sentence="רותפל ונילעש היעב וז."
ground_truth1 = "this is a problem we have to solve."
ground_truth2 = "this is a problem we've got to solve."
ground_truth3 = "this is a problem we need to solve."
ground_truth4 = "this is a problem we must solve."
ground_truth5 = "it's a problem we've got to solve."
ground_truth6 = "it's a problem we need to solve."
ground_truth7 = "it's a problem we have to solve."
ground_truth8 = "it's a problem we must solve."
translated_text, translated_tokens, attention_weights = evaluate(sentence)
#print_translation(sentence, translated_text, ground_truth)
reference = [ground_truth1, ground_truth2, ground_truth3, ground_truth4, ground_truth5, 
ground_truth6, ground_truth7, ground_truth8]
candidate = translated_text.numpy().decode("utf-8")
candidate = candidate.replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")1004  .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")1005  .replace(" ) ", ") ").replace(" , ", ", ")
print("candidate: " + candidate)
print('Transformer METEOR score -> {}'.format(ms.meteor_score(reference, candidate)))
candidate = Translator().translate(sentence, src = 'he').text.lower()
print("Google Translator: " + candidate)
print('Google Translate METEOR score -> {}'.format(ms.meteor_score(reference, candidate)))
# Second test
sentence="תונחל יתכלהז."
ground_truth1 = "i went to the store."
ground_truth2 = "i went to the shop."
translated_text, translated_tokens, attention_weights = evaluate(sentence)
#print_translation(sentence, translated_text, ground_truth1)
reference = [ground_truth1, ground_truth2]
candidate = translated_text.numpy().decode("utf-8")
candidate = candidate.replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")1024  .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")1025  .replace(" ) ", ") ").replace(" , ", ", ")
print("candidate: " + candidate)
print('Transformer METEOR score -> {}'.format(ms.meteor_score(reference, candidate)))
candidate = Translator().translate(sentence, src = 'he').text.lower()
print("Google Translator: " + candidate)
print('Google METEOR score -> {}'.format(ms.meteor_score(reference, candidate)))
# Third test
sentence="דלי לדגל ידכרפכ ךירצ."
ground_truth1 = "it takes a village to raise a child."
ground_truth2 = "it takes a village to raise a kid."
ground_truth3 = "it takes a town to raise a child."1038 ground_truth4 = "it takes a town to raise a kid."
ground_truth5 = "we need a village to raise a child."
ground_truth6 = "we need a village to raise a kid."
ground_truth7 = "we need a town to raise a child."
ground_truth8 = "we need a town to raise a kid."
ground_truth9 = "you need a village to raise a child."
ground_truth10 = "you need a village to raise a kid."
ground_truth11 = "you need a town to raise a child."
ground_truth12 = "you need a town to raise a kid."
translated_text, translated_tokens, attention_weights = evaluate(sentence)
#print_translation(sentence, translated_text, ground_truth)
reference = [ground_truth1, ground_truth2, ground_truth3, ground_truth4, ground_truth5, 
ground_truth6, ground_truth7, ground_truth8, ground_truth9, ground_truth10, ground_truth11, 
ground_truth12]
candidate = translated_text.numpy().decode("utf-8")
candidate = candidate.replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")1056  .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")1057  .replace(" ) ", ") ").replace(" , ", ", ")
print("candidate: " + candidate)
print('Transformer METEOR score -> {}'.format(ms.meteor_score(reference, candidate)))
candidate = Translator().translate(sentence, src = 'he').text.lower()
print("Google Translator: " + candidate)
print('Google METEOR score -> {}'.format(ms.meteor_score(reference, candidate)))
# Create visualization of Transformer internal operation
def plot_attention_head(in_tokens, translated_tokens, attention):
# The plot is of the attention when a token was generated.
# The model didn't generate `<START>` in the output. Skip it.
translated_tokens = translated_tokens[1:]
ax = plt.gca()
ax.matshow(attention)
ax.set_xticks(range(len(in_tokens)))
ax.set_yticks(range(len(translated_tokens)))
labels = [label.decode('utf-8') for label in in_tokens.numpy()]
ax.set_xticklabels(
labels, rotation=90)
labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
ax.set_yticklabels(labels)
head = 0
# shape: (batch=1, num_heads, seq_len_q, seq_len_k)
attention_heads = tf.squeeze(
attention_weights['decoder_layer4_block2'], 0)
attention = attention_heads[head]
attention.shape
# Tokenize last Hebrew input sentence
in_tokens = tf.convert_to_tensor([sentence])
in_tokens = tokenizers.he.tokenize(in_tokens).to_tensor()
in_tokens = tokenizers.he.lookup(in_tokens)[0]1094 np.char.decode(in_tokens.numpy().astype(np.bytes_), 'UTF-8')
# View translated tokens
translated_tokens
# Visualize Transformer attention mechanism operation
plot_attention_head(in_tokens, translated_tokens, attention)
# Code to plot individual attention weights
def plot_attention_weights(sentence, translated_tokens, attention_heads):
in_tokens = tf.convert_to_tensor([sentence])
in_tokens = tokenizers.he.tokenize(in_tokens).to_tensor()
in_tokens = tokenizers.he.lookup(in_tokens)[0]
in_tokens
fig = plt.figure(figsize=(16, 8))
for h, head in enumerate(attention_heads):
ax = fig.add_subplot(2, 4, h+1)
plot_attention_head(in_tokens, translated_tokens, head)
ax.set_xlabel(f'Head {h+1}')
plt.tight_layout()
plt.show()
# Plot the 8 attention heads in decoder layer 4 for last Hebrew input
plot_attention_weights(sentence, translated_tokens,
attention_weights['decoder_layer4_block2'][0])
# Run basic test on 10 sentences taken from input corpus but not used in training
f= open("test.he", "r")
he_tests = f.readlines()
f.close()
f= open("test.en", "r")
en_tests = f.readlines()
f.close()
length = len(he_tests)
meteor_total = 0
for i in range(length):
he_tests[i] = he_tests[i][:-1]
en_tests[i] = en_tests[i][:-1]
reference = en_tests[i].lower()
print(str(i) + ". Reference: " + reference)
reference = [reference]
translated_text, translated_tokens, attention_weights = evaluate(he_tests[i])
candidate = translated_text.numpy().decode("utf-8")
candidate = candidate.replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")1144  .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")1145  .replace(" ) ", ") ").replace(" , ", ", ")
print(str(i) + ". Transformer translation: " + candidate)
meteor = ms.meteor_score(reference, candidate)
print(str(i) + ". METEOR: " + str(meteor))
meteor_total += meteor1150
print('Transformer METEOR score -> {}'.format(meteor_total / length))
meteor_total = 0
# Compare to Google Translate
for i in range(length):
reference = en_tests[i].lower()
print(str(i) + ". Reference: " + reference)
reference = [reference]
candidate = Translator().translate(he_tests[i], src = 'he').text.lower()
print(str(i) + ". Google Translator: " + candidate)
meteor = ms.meteor_score(reference, candidate)
print(str(i) + ". Google METEOR: " + str(meteor))
meteor_total += meteor
print('Google METEOR score -> {}'.format(meteor_total / length))
# Now run a more extended test on 1000 unseen sentences from input corpus
print(time.time())
f= open("test_1000.he", "r")
he_tests = f.readlines()
f.close()
f= open("test_1000.en", "r")
en_tests = f.readlines()
f.close()
length = len(he_tests)
meteor_total = 0
for i in range(length):
he_tests[i] = he_tests[i][:-1]
en_tests[i] = en_tests[i][:-1]
reference = en_tests[i].lower()
#print(str(i) + ". Reference: " + reference)
reference = [reference]
translated_text, translated_tokens, attention_weights = evaluate(he_tests[i])
candidate = translated_text.numpy().decode("utf-8")
candidate = candidate.replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")1188  .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")1189  .replace(" ) ", ") ").replace(" , ", ", ")
#print(str(i) + ". Transformer translation: " + candidate)
meteor = ms.meteor_score(reference, candidate)
#print(str(i) + ". METEOR: " + str(meteor))
meteor_total += meteor
if i % 100 == 0:
print(str(i) + ": " + str(time.time()))
print('Transformer METEOR score -> {}'.format(meteor_total / length))
meteor_total = 0
# Again compare to Google Translate
for i in range(length):
reference = en_tests[i].lower()
#print(str(i) + ". Reference: " + reference)1206  reference = [reference]
candidate = Translator().translate(he_tests[i], src = 'he').text.lower()
#print(str(i) + ". Google Translator: " + candidate)
meteor = ms.meteor_score(reference, candidate)
#print(str(i) + ". Google METEOR: " + str(meteor))
meteor_total += meteor
if i % 100 == 0:
print(str(i) + ": " + str(time.time()))
print('Google METEOR score -> {}'.format(meteor_total / length))
# Finally, a very challenging test: translate the first chapter of Genesis from the Hebrew Bible
f= open("bibtest.he", "r")
he_tests = f.readlines()
f.close()
f= open("bibtest.en", "r")
en_tests = f.readlines()
f.close()
length = len(he_tests)
meteor_total = 0
for i in range(length):
he_tests[i] = he_tests[i][:-1]
en_tests[i] = en_tests[i][:-1]
reference = en_tests[i].lower()
print(str(i) + ". Reference: " + reference)
reference = [reference]
translated_text, translated_tokens, attention_weights = evaluate(he_tests[i])
candidate = translated_text.numpy().decode("utf-8")
candidate = candidate.replace(" ' ", "'").replace(" .", ".").replace(" ?", "?").replace(" !", "!")1237  .replace(' " ', '" ').replace(' "', '"').replace(" : ", ": ").replace(" ( ", " (")1238  .replace(" ) ", ") ").replace(" , ", ", ")
print(str(i) + ". Transformer translation: " + candidate)
meteor = ms.meteor_score(reference, candidate)
print(str(i) + ". METEOR: " + str(meteor))
meteor_total += meteor
print('Transformer METEOR score -> {}'.format(meteor_total / length))
meteor_total = 0
# Compare to Google Translate
for i in range(length):
reference = en_tests[i].lower()
print(str(i) + ". Reference: " + reference)
reference = [reference]
candidate = Translator().translate(he_tests[i], src = 'he').text.lower()
print(str(i) + ". Google Translator: " + candidate)
meteor = ms.meteor_score(reference, candidate)
print(str(i) + ". Google METEOR: " + str(meteor))
meteor_total += meteor
print('Google METEOR score -> {}'.format(meteor_total / length))