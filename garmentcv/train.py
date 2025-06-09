import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist

def load_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train[..., None] / 255.0  # shape = (N,28,28,1)
    x_test  = x_test[..., None]  / 255.0
    return (x_train, y_train), (x_test, y_test)

def build_custom_cnn():
    inputs = layers.Input(shape=(28,28,1))
    
    # Data augmentation
    x = layers.RandomTranslation(0.1,0.1)(inputs)
    x = layers.RandomRotation(0.1)(x)
    
    # Block 1
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 2
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 3
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)
    
    # Head
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_custom_cnn()
    model.summary()
    model.fit(x_train, y_train,
              validation_split=0.1,
              epochs=20, batch_size=64)
    loss, acc = model.evaluate(x_test, y_test)
    print("Test accuracy:", acc)
    model.save('fashion_custom_cnn.h5')

if __name__=='__main__':
    main()
