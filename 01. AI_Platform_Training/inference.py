import pandas as pd
import numpy as np
import tensorflow as tf
path_dataset = "/Users/macbook/Desktop/Rabbit/00. Preprocessing_dataset/test_census.csv"
df = pd.read_csv(path_dataset)
x = df.drop(["Unnamed: 0","income_bracket","dataframe"],axis=1)
y = df[["income_bracket"]]

data = list(x.iloc[0])

model = tf.keras.models.load_model("models")
model.summary()

correct = np.expand_dims(data,0)

model.predict([data])