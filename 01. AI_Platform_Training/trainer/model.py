#google-cloud-storage
#pandas
#numpy
#tensorflow
from google.cloud import storage
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
#----------------------------------------------------------------
#----------------------------------------------------------------

def download_ds(BUCKET, FILE, KEY=None):
    # instancio cliente
    try:
        client_storage = storage.Client()
    except:
        client_storage = storage.Client.from_service_account_json(KEY)
    bucket = client_storage.get_bucket(BUCKET)
    blob = bucket.blob(FILE)
    # descargo archivo en el folder actual
    blob.download_to_filename(FILE)

def get_features(df):
    x = np.array(df.drop(["income_bracket","dataframe"], axis=1))
    y = np.array(df["income_bracket"])
    return x,y

def load_data(BUCKET,FILE,KEY=None):
    download_ds(BUCKET,FILE,KEY)
    df = pd.read_csv(FILE)
    x,y = get_features(df)
    return x,y

def build_model(DENSE_UNITS=32):
    inputs = Input(shape=(80,))
    h = Dense(units=DENSE_UNITS, activation='relu')(inputs)
    h = Dropout(0.5)(h)
    h = Dense(units=DENSE_UNITS, activation='relu')(h)
    h = Dropout(0.5)(h)
    outputs = Dense(units=1, activation='sigmoid')(h)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                loss= "binary_crossentropy",
                metrics=['accuracy'])
    return model

def train_and_evaluate(args):
    model = build_model(args["DENSE_UNITS"])
    print(model.summary())

    print("\n")
    print("Prepare dataset ...")
    
    try:
        KEY = args["KEY"]
    except:
        KEY = None

    # load data train
    x_train, y_train = load_data(args["BUCKET"], args["FILE_TRAIN"], KEY)

    # load data validation
    x_val, y_val = load_data(args["BUCKET"], args["FILE_VAL"], KEY)

    # define callbacks
    early_stopping = EarlyStopping(monitor="val_loss",patience=5)

    history = model.fit(
                x=x_train,
                y=y_train,
                validation_data=(x_val,y_val),
                epochs=args["EPOCHS"],
                batch_size=args["BATCH_SIZE"],
                callbacks=[early_stopping]
    )
    # save model    
    tf.saved_model.save(
        obj=model, export_dir=args["OUTPUT_DIR"])
    print(f"Exported trained model to {args['OUTPUT_DIR']}")

