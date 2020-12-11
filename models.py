from tensorflow import keras

def get_compiled_model():
  model = keras.Sequential([
    keras.layers.Dense(12, activation='relu'),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(48, activation='relu'),
    keras.layers.Dense(4)
  ])

  model.compile(optimizer='adam',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model
