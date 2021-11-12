from datasets import load_dataset
from tensorflow import keras
import time
import json
import signal
import sys
import pdb

gpu_type = sys.argv[1]
raw_datasets = load_dataset("imdb")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

#inputs = tokenizer(sentences, padding="max_length", truncation=True)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(800)) # each inf epoch includes 100 runs
full_eval_dataset = tokenized_datasets["test"]

import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification
from datetime import datetime
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

tf_eval_dataset = small_eval_dataset.remove_columns(["text"]).with_format("tensorflow")

eval_features = {x: tf_eval_dataset[x].to_tensor() for x in tokenizer.model_input_names}

eval_tf_dataset = tf.data.Dataset.from_tensor_slices((eval_features, tf_eval_dataset["label"]))
eval_tf_dataset = eval_tf_dataset.batch(8)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=tf.metrics.SparseCategoricalAccuracy(),
)

global curr_iter, batch_time
batch_time = {} # mini-batch time
curr_iter = 0

class RecordBatch(keras.callbacks.Callback):
    def __init__(self):
        super(RecordBatch, self).__init__()
        global curr_iter, batch_time
        self.batch_time = []
        self.batch_begin = 0
        self.curr_iter = curr_iter
    def on_predict_batch_begin(self, batch, logs=None):
        self.batch_begin = time.time()
    def on_predict_batch_end(self, batch, logs=None):
        self.batch_time.append(round(time.time() - self.batch_begin, 4))
    def on_predict_end(self, logs=None):
        global curr_iter, batch_time
        batch_time[curr_iter] = self.batch_time

my_callback = RecordBatch()
callbacks = [my_callback]

################### connects interrupt signal to the process #####################

def terminateProcess(signalNumber, frame):
    # first record the wasted epoch time
    global batch_time
    with open(f'logs/{gpu_type}_bert.json', 'w') as f:
        json.dump(batch_time, f, indent=4)
    sys.exit()

signal.signal(signal.SIGINT, terminateProcess)

#################################################################################

print('################## start inferencing ##################')

i = 0
while True:
    i += 1
    curr_iter = i
    batch_time[i] = []
    for data in eval_tf_dataset:
        start_time = time.time()
        model.predict_on_batch(data)
        duration = round(time.time() - start_time,3)
        batch_time[i].append(duration)

