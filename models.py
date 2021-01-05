from tensorflow import keras

def get_compiled_model(learning_rate=0.01):
  model = keras.Sequential([
    keras.layers.Dense(12, activation='relu'),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(48, activation='relu'),
    keras.layers.Dense(4)
  ])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

def variable_model(learning_rate=0.01, n_layers=3, n_neurons=24):
    model = keras.Sequential()
    for i in range (n_layers):
        model.add(keras.layers.Dense(n_neurons, activation='relu'))

    model.add(keras.layers.Dense(4))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model
