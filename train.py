import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from model import create_cnn_lstm

SEQ_LEN = 10

data = []
labels = []

label_map = {}
label_id = 0

for folder in os.listdir("dataset"):
    label_map[label_id] = folder

    images = os.listdir(f"dataset/{folder}")

    # create sequences
    for i in range(len(images) - SEQ_LEN):
        seq = []
        for j in range(SEQ_LEN):
            img = cv2.imread(f"dataset/{folder}/{images[i+j]}")
            img = cv2.resize(img, (64,64))
            seq.append(img)

        data.append(seq)
        labels.append(label_id)

    label_id += 1

data = np.array(data) / 255.0
labels = to_categorical(labels)

model = create_cnn_lstm(len(label_map))

model.fit(data, labels, epochs=10)

model.save("cnn_lstm_model.h5")