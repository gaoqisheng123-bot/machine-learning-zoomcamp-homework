import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import os

BEST_SIZE = 256
BEST_DROP = 0.2
BEST_LR = 0.001
EPOCHS = 20
IMG_SIZE = (32, 32)

def train_final_model():
    print("Loading and preparing data...")
    df = pd.read_csv('data/Train.csv')
    df['Path'] = 'data/' + df['Path']
    df['ClassId'] = df['ClassId'].astype(int)
    
    train_set, val_set = train_test_split(df, test_size=0.2, stratify=df['ClassId'], random_state=42)
    
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(df['ClassId']), y=df['ClassId'])
    weights_dict = {i: w for i, w in enumerate(weights)}
    
    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=10, width_shift_range=0.1, 
        height_shift_range=0.1, brightness_range=[0.7, 1.3]
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow_from_dataframe(train_set, x_col='Path', y_col='ClassId', target_size=IMG_SIZE, class_mode='raw', batch_size=32)
    val_gen = val_datagen.flow_from_dataframe(val_set, x_col='Path', y_col='ClassId', target_size=IMG_SIZE, class_mode='raw', batch_size=32)

    print("Building model...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(BEST_SIZE, activation='relu'),
        layers.Dropout(BEST_DROP),
        layers.Dense(43, activation='softmax')
    ])
    
    model.compile(optimizer=optimizers.Adam(learning_rate=BEST_LR), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("Starting training...")
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, class_weight=weights_dict)
    
    model.save('traffic_sign_model.h5')
    print("Success! Model saved as traffic_sign_model.h5")

if __name__ == "__main__":
    train_final_model()
