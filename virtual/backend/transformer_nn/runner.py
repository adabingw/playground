import tensorflow as tf

import tensorflow_text

from transformer_nn.transformer import Transformer
from transformer_nn.translator import Translator

import warnings

import os 

warnings.simplefilter(action="ignore")

MAX_TOKENS=128
BUFFER_SIZE = 20000
BATCH_SIZE = 64

NUM_LAYERS = 4
D_MODEL = 128
DFF = 512
NUM_HEADS = 8
DROPOUT_RATE = 0.1

def print_translation(sentence, tokens, ground_truth = "no ground truth provided"):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
    print(f'{"Ground truth":15s}: {ground_truth}')
    
def new_translator(): 
    tokenizers = tf.saved_model.load('transformer_nn/ted_hrlr_translate_pt_en_converter')
    model = Transformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
        target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
        dropout_rate=DROPOUT_RATE)
    
    checkpoint_path = './checkpoints/model'
    model.load_weights(checkpoint_path)
    
    translator = Translator(tokenizers, model)
    return translator

def translate(translator, sentence, ground_truth): 
    translated_text, translated_tokens, attention_weights = translator(
        tf.constant(sentence))
    print_translation(sentence, translated_text, ground_truth)