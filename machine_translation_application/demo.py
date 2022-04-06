from __future__ import absolute_import, division, print_function, unicode_literals
from multiprocessing.dummy import active_children
from pyexpat import model
import re
from matplotlib.pyplot import sca
import tensorflow as tf
tf.enable_eager_execution()
import tensorflow_datasets as tfds
import numpy as np
import os

def split_task_name(task_name):
    task_name_split = task_name.split('_to_')
    return task_name_split[0], task_name_split[1]


# Get dataset
def get_ted_hrlr_translate_dataset(task_name="pt_to_en", BATCH_SIZE=64, MAX_LENGTH=40, BUFFER_SIZE=20000, 
                                    languageA_target_vovab_size=2**13, languageB_target_vovab_size=2**13):
    """
    task_name: string
    BATCH_SIZE: int
    MAX_LENGTH: int
    BUFFER_SIZE: int
    languageA_target_vacab_size: int
    languageB_target_vacab_size: int
    """
    task_name_prefix = 'ted_hrlr_translate'
    task_name_list = ['az_to_en', 'az_tr_to_en', 'be_to_en',
                      'be_ru_to_en', 'es_to_pt', 'fr_to_pt',
                      'gl_to_en', 'gl_pt_to_en', 'he_to_pt',
                      'it_to_pt', 'pt_to_en', 'ru_to_en',
                      'ru_to_pt', 'tr_to_en']
    if task_name not in task_name_list:
        raise ValueError(f'Choose task_name from {task_name_list}')

    complete_task_name = task_name_prefix + '/' + task_name
    # Get the dataset
    example, metadata = tfds.load(complete_task_name, with_info=True, as_supervised=True)
    train_examples, val_examples = example['train'], example['validation']

    # make dir to store data
    if not os.path.exists(complete_task_name):
        os.makedirs(complete_task_name)

    # load data and encode the string as int
    tokenizer_languageA_path = os.path.join(complete_task_name, split_task_name(task_name)[0])
    tokenizer_languageA_complete_path = tokenizer_languageA_path + ".subwords"
    if not os.path.exists(tokenizer_languageA_complete_path):
        tokenizer_languageA = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (languageA.numpy() for languageA, _ in train_examples), 
            trarget_vocab_size=languageA_target_vovab_size)
        tokenizer_languageA.save_to_file(tokenizer_languageA_path)
    else:
        tokenizer_languageA = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizer_languageA_path)

    tokenizer_languageB_path = os.path.join(complete_task_name, split_task_name(task_name)[1])
    tokenizer_languageB_complete_path = tokenizer_languageB_path + ".subwords"
    if not os.path.exists(tokenizer_languageB_complete_path):
        tokenizer_languageB = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (languageB.numpy() for _, languageB in train_examples), 
            trarget_vocab_size=languageB_target_vovab_size)
        tokenizer_languageB.save_to_file(tokenizer_languageB_path)
    else:
        tokenizer_languageB = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizer_languageB_path)

    def encode(lang1, lang2):
        lang1 = [tokenizer_languageA.vocab_size] + tokenizer_languageA.encode(
            lang1.numpy()) + [tokenizer_languageA.vocab_size + 1]
        lang2 = [tokenizer_languageB.vocab_size] + tokenizer_languageB.encode(
            lang2.numpy()) + [tokenizer_languageB.vocab_size + 1]
        return lang1, lang2

    def filter_max_length(x, y, max_length = MAX_LENGTH):
        return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)

    def tf_encode(pt, en):
        return tf.py_function(encode, [pt, en], [tf.int32, tf.int32])
    
    train_dataset = train_examples.map(tf_encode)
    train_dataset = train_dataset.filter(filter_max_length)
     # cache the dataset to memory to get a speedup while reading from it.
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)   # <<PrefetchDataset shapes: ((?, ?), (?, ?)), types: (tf.int32, tf.int32)>

    val_dataset = val_examples.map(tf_encode)
    val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
    
    return train_dataset, val_dataset


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], 
                            np.arange(d_model)[np.newaxis, :], d_model)
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]    # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask # (seq_len, seq_len)  lower_triangle_matrix

# Attention
def scaled_dot_product_attention(q, k, v, mask=None):
    '''
    q, k, v mush have matching leading dimension
    k, v must have matching penultimate dimension, i.e. seq_len_k = seq_len_v
    q: (..., seq_len_q, depth)
    k: (..., seq_len_k, depth)
    v: (..., seq_len_v, depth_v)
    mask: shape broadcastable to (..., seq_len_q, seq_len_k)
    return:
    output, attention_weights
    '''
    matmul_qk = tf.matmul(q, k, transpose_b=True) # (..., seq_len_q, seq_len_k)

    #scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add mask
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)    # (..., seq_len_v, depth_v)
    return output, attention_weights

# multi head  d_model = depth * heads_num
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert self.d_model % self.num_heads == 0
        self.depth = self.d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)


    def split_heads(self, x, batch_size):
        '''
        spilt the last dimension into (num_heads, depth)
        Transpose the result into (batch_size, num_heads, seq_len, depth)
        '''
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wq(k)
        v = self.wq(v)

        q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])    #(batch_size, seq_len_v, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model)) # (batch_size, seq_len_v, d_model)

        out_put = self.dense(concat_attention)  #(batch_size, seq_len_v, d_model)
        return out_put, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
        ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormaliaztion(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNoemalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(ffn_output + out1)
        return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        
        return out3, attn_weights_block1, attn_weights_block2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()
        

if __name__ == "__main__":
    ans = MultiHeadAttention(4, 2)
    