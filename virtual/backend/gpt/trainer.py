from datasets import load_dataset, Dataset, DatasetDict
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelWithLMHead
from tensorflow.keras.optimizers import Adam

train_percentage = 0.9
validation_percentage = 0.07
test_percentage = 0.03

# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)

# load dataset
datasets = load_dataset("huggingartists/taylor-swift")

# test

train, validation, test = np.split(datasets['train']['text'], [int(len(datasets['train']['text'])*train_percentage), int(len(datasets['train']['text'])*(train_percentage + validation_percentage))])

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenized_data = tokenizer(datasets["sentence"], return_tensors="np", padding=True)
# Tokenizer returns a BatchEncoding, but we convert that to a dict for Keras
tokenized_data = dict(tokenized_data)

labels = np.array(datasets["label"])  # Label is already an array of 0 and 1

# for i in range(7): 
#     print(tokenized_data[i], labels[i])

print(tokenized_data)
print(labels)

# datasets = DatasetDict(
#     {
#         'train': Dataset.from_dict({'text': list(train)}),
#         'validation': Dataset.from_dict({'text': list(validation)}),
#         'test': Dataset.from_dict({'text': list(test)})
#     }
# )

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
model.compile(optimizer=Adam(3e-5))
# model.fit(tokenized_data, labels)
