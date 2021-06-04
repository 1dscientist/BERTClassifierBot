import os
from flask import Flask, render_template
from flask import request

import tensorflow as tf
from urllib.request import urlretrieve
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from tensorflow.keras.layers import Dropout, Dense

model_name = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)

SNIPS_DATA_BASE_URL = (
    "https://github.com/ogrisel/slot_filling_and_intent_detection_of_SLU/blob/"
    "master/data/snips/"
)
for filename in ["train", "valid", "test", "vocab.intent", "vocab.slot"]:
    path = Path(filename)
    if not path.exists():
        print(f"Downloading {filename}...")
        urlretrieve(SNIPS_DATA_BASE_URL + filename + "?raw=true", path)


def parse_line(line):
    data, intent_label = line.split(" <=> ")
    items = data.split()
    words = [item.rsplit(":", 1)[0] for item in items]
    word_labels = [item.rsplit(":", 1)[1] for item in items]
    return {
        "intent_label": intent_label,
        "words": " ".join(words),
        "word_labels": " ".join(word_labels),
        "length": len(words),
    }


def encode_dataset(text_sequences):
    # Create token_ids array (initialized to all zeros), where
    # rows are a sequence and columns are encoding ids
    # of each token in given sequence.
    token_ids = np.zeros(shape=(len(text_sequences), max_token_len),
                         dtype=np.int32)

    for i, text_sequence in enumerate(text_sequences):
        encoded = tokenizer.encode(text_sequence)
        token_ids[i, 0:len(encoded)] = encoded

    attention_masks = (token_ids != 0).astype(np.int32)
    return {"input_ids": token_ids, "attention_masks": attention_masks}


train_lines = Path("train").read_text().strip().splitlines()
valid_lines = Path("valid").read_text().strip().splitlines()
test_lines = Path("test").read_text().strip().splitlines()

df_train = pd.DataFrame([parse_line(line) for line in train_lines])
df_valid = pd.DataFrame([parse_line(line) for line in valid_lines])
df_test = pd.DataFrame([parse_line(line) for line in test_lines])

max_token_len = 43

encoded_train = encode_dataset(df_train["words"])
encoded_valid = encode_dataset(df_valid["words"])
encoded_test = encode_dataset(df_test["words"])

intent_names = Path("vocab.intent").read_text().split()
intent_map = dict((label, idx) for idx, label in enumerate(intent_names))
intent_train = df_train["intent_label"].map(intent_map).values
intent_valid = df_valid["intent_label"].map(intent_map).values
intent_test = df_test["intent_label"].map(intent_map).values

slot_names = ["[PAD]"] + Path("vocab.slot").read_text().strip().splitlines()
slot_map = {}
for label in slot_names:
    slot_map[label] = len(slot_map)

class JointIntentAndSlotFillingModel(tf.keras.Model):

    def __init__(self, intent_num_labels=None, slot_num_labels=None,
                 dropout_prob=0.1):
        super().__init__(name="joint_intent_slot")

        self.bert = base_bert_model

        # TODO: define the dropout, intent & slot classifier layers
        ### YOUR CODE HERE ###
        # Solution:
        self.dropout = Dropout(dropout_prob)
        self.intent_classifier = Dense(intent_num_labels,
                                       name="intent_classifier")
        self.slot_classifier = Dense(slot_num_labels,
                                     name="slot_classifier")

    def call(self, inputs, **kwargs):
        # Extract features from the inputs using pre-trained BERT.
        # TODO: what does the bert model return?
        tokens_output, pooled_output = self.bert(inputs, **kwargs, return_dict=False)

        # TODO: use the new layers to predict slot class (logits) for each
        # token position in input sequence (size: (batch_size, seq_len, slot_num_labels)).
        ### YOUR CODE HERE ###
        # Solution:
        tokens_output = self.dropout(tokens_output,
                                     training=kwargs.get("training", False))
        slot_logits = self.slot_classifier(tokens_output)

        # TODO: define a second classification head for the sequence-wise
        # predictions (size: (batch_size, intent_num_labels)).
        # (Hint: create pooled_output to get the intent_logits).
        # Remember that the 2nd output of the main BERT layer is size
        # (batch_size, output_dim) & gives a "pooled" representation for
        # full sequence from hidden state corresponding to [CLS]).
        ### YOUR CODE HERE ###
        # Solution:
        pooled_output = self.dropout(pooled_output,
                                     training=kwargs.get("training", False))
        intent_logits = self.intent_classifier(pooled_output)

        return slot_logits, intent_logits


# TODO: create an instantiation of this model
joint_model = JointIntentAndSlotFillingModel( \
    intent_num_labels=len(intent_map), slot_num_labels=len(slot_map))

def show_predictions(text, intent_names, slot_names):
    inputs = tf.constant(tokenizer.encode(text))[None, :]  # batch_size = 1
    outputs = joint_model(inputs)
    slot_logits, intent_logits = outputs
    slot_ids = slot_logits.numpy().argmax(axis=-1)[0, 1:-1]
    intent_id = intent_logits.numpy().argmax(axis=-1)[0]
    print("## Intent:", intent_names[intent_id])
    print("## Slots:")
    for token, slot_id in zip(tokenizer.tokenize(text), slot_ids):
        print(f"{token:>10} : {slot_names[slot_id]}")

def answer_question(question):
    return show_predictions(question, intent_names, slot_names)


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
  
    if request.method == 'POST':
      form = request.form
      result = []
      bert_abstract = form['paragraph']
      question = form['question']
      result.append(form['question'])
      result.append(answer_question(question))
      result.append(form['paragraph'])

    return render_template("index.html",result = result)
    return render_template("index.html")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
