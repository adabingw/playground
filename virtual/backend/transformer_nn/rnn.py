import tensorflow_datasets as tfds
import tensorflow as tf

from transformer_nn.trainer import CustomSchedule, masked_accuracy, masked_loss
from transformer_nn.transformer import Transformer
from transformer_nn.translator import Translator

MAX_TOKENS=128
BUFFER_SIZE = 20000
BATCH_SIZE = 64

NUM_LAYERS = 4
D_MODEL = 128
DFF = 512
NUM_HEADS = 8
DROPOUT_RATE = 0.1

def train(): 

    # downloads the dataset
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                with_info=True,
                                as_supervised=True)

    train_examples, val_examples = examples['train'], examples['validation']

    # checking the data
    for pt_examples, en_examples in train_examples.batch(3).take(1):
        print('> Examples in Portuguese:')
        for pt in pt_examples.numpy():
            print(pt.decode('utf-8'))
        print()

        print('> Examples in English:')
        for en in en_examples.numpy():
            print(en.decode('utf-8'))

    # setting up tokenizer
    # utilizes subword tokenizing implementation
    model_name = 'ted_hrlr_translate_pt_en_converter'

    tf.keras.utils.get_file(
        f'{model_name}.zip',
        f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
        cache_dir='.', cache_subdir='', extract=True
    )
    tokenizers = tf.saved_model.load(model_name)

    # takes batches of text as input, and converts them to a format suitable for training
    def prepare_batch(pt, en):
        pt = tokenizers.pt.tokenize(pt)         # Output is ragged.
        pt = pt[:, :MAX_TOKENS]                 # Trim to MAX_TOKENS.
        pt = pt.to_tensor()                     # Convert to 0-padded dense Tensor

        # at each input location the label is the id of the next token.
        en = tokenizers.en.tokenize(en)
        en = en[:, :(MAX_TOKENS+1)]
        en_inputs = en[:, :-1].to_tensor()      # Drop the [END] tokens
        en_labels = en[:, 1:].to_tensor()       # Drop the [START] tokens

        # splits the target (English) tokens into inputs and labels.
        # output format is (inputs, labels)
        return (pt, en_inputs), en_labels

    # converts dataset of text examples into data of batches for training 
    def make_batches(ds):
        return (
            ds
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(prepare_batch, tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE))
    
    # Create training and validation set batches.
    train_batches = make_batches(train_examples)
    val_batches = make_batches(val_examples)
        
    transformer = Transformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
        target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
        dropout_rate=DROPOUT_RATE)

    learning_rate = CustomSchedule(D_MODEL)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                        epsilon=1e-9)

    transformer.compile(
        loss=masked_loss,
        optimizer=optimizer,
        metrics=[masked_accuracy])

    transformer.fit(train_batches,
                    epochs=8,
                    validation_data=val_batches)
    
    # transformer.save('transformer.model')
    transformer.save_weights('./checkpoints/checkpoint')