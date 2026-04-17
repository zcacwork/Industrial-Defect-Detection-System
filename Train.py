import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

 
train_path = 'Data/casting_data/train'
test_path = 'casting_data/test'

 
train_datagen = ImageDataGenerator(
    rescale=1./255,         
    rotation_range=20,        
    horizontal_flip=True,     
    validation_split=0.2      
)

test_datagen = ImageDataGenerator(rescale=1./255)

 
train_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=(300, 300),   
    batch_size=32,
    class_mode='binary',      
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

print(f"Class Indices: {train_gen.class_indices}")