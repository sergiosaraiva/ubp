import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Masking
from tensorflow.keras.callbacks import EarlyStopping

class ubp:
    def __init__(self, config):
        self.config = config
        self.model = None
        # Assuming max length from config, or set a default
        self.max_length = config.get('max_sequence_length', 7)

    def preprocess_data(self, file_path, return_labels=True):
        data = pd.read_csv(file_path)
        data.fillna(0, inplace=True)  # Assuming padding with 0
        X = data.iloc[:, :-1].values
        X = np.expand_dims(X, axis=-1)  # Adding feature dimension

        if return_labels:
            y = data.iloc[:, -1].astype(int)
            num_classes = np.max(y) + 1  # Dynamically find number of classes
            y = to_categorical(y, num_classes=num_classes)
            return X, y
        return X

    def build_model(self):
        model = Sequential()
        model.add(Masking(mask_value=0, input_shape=(self.max_length, 1)))  # Mask zero values
        model.add(LSTM(self.config['layer_configurations'][0]['units'],
                    activation=self.config['layer_configurations'][0]['activation'], return_sequences=self.config['layer_configurations'][0]['return_sequences']))
        model.add(Dense(self.config['layer_configurations'][1]['units'], activation=self.config['layer_configurations'][1]['activation']))
        
        model.compile(optimizer=self.config['optimizer'], loss=self.config['loss'])
        self.model = model

    def train_model(self, X, y, epochs, batch_size):
        # Check if early stopping is applied
        if self.config.get('early_stopping', {}).get('apply_early_stopping', False):
            es_config = self.config['early_stopping']
            early_stopping = EarlyStopping(
                monitor=es_config.get('monitor', 'val_loss'),
                min_delta=es_config.get('min_delta', 0.001),
                patience=es_config.get('patience', 10),
                verbose=es_config.get('verbose', 1),
                mode=es_config.get('mode', 'min'),
                restore_best_weights=es_config.get('restore_best_weights', True)
            )
            callbacks = [early_stopping]
        else:
            callbacks = []
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_split=0.1  # Ensure there's a validation split if using early stopping
        )

    def predict_next_steps(self, X):
        try:
            predictions = self.model.predict(X)
            top_k = self.config.get('top_k_predictions', 3)  # Default to 3 if not specified
            # Get top 'top_k' predictions for each sample
            top_predictions = np.argsort(-predictions, axis=1)[:, :top_k]  # Sort predictions in descending order
            top_probs = np.sort(predictions, axis=1)[:, -top_k:][:, ::-1]  # Sort probabilities in descending order

            # Create a structured result to return
            results = []
            for preds, probs in zip(top_predictions, top_probs):
                results.append({'predictions': preds, 'probabilities': probs})

            return results
        except Exception as e:
            print("Failed to predict:", e)
            return []  # Return an empty list in case of any error


    def save_model(self, file_path):
        self.model.save(file_path + ".keras")

    def load_model(self, file_path):
        complete_path = file_path + ".keras"
        if os.path.exists(complete_path):
            self.model = load_model(complete_path)
        else:
            raise FileNotFoundError(f"No model found at {complete_path}")
