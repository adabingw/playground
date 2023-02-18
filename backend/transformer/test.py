import tensorflow as tf 



def print_translation(sentence, tokens, ground_truth):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
    print(f'{"Ground truth":15s}: {ground_truth}')
    
def test(): 
    sentence = 'este é um problema que temos que resolver.'
    ground_truth = 'this is a problem we have to solve .'

    translated_text, translated_tokens, attention_weights = translator(
        tf.constant(sentence))
    print_translation(sentence, translated_text, ground_truth)