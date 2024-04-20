import tensorflow as tf
shakespeare_url = "http://homl.info/shakespeare"
filepath = tf.keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_txt = f.read()
text_vec_layer = tf.keras.layers.TextVectorization(split="character", standardize='lower')
text_vec_layer.adapt([shakespeare_txt])
encoded = text_vec_layer([shakespeare_txt])[0]
encoded -= 2
n_tokens = text_vec_layer.vocabulary_size() -2
dataset_size= len(encoded)

def to_dataset(sequence, length, shuffle=False, seed=None, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length + 1))
    if shuffle:
        ds = ds.shuffle(buffer_size=100_100, seed=seed)
    ds = ds.repeat().batch(batch_size)  # Add .repeat() here
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)


length = 100
tf.random.set_seed(42)
train_set = to_dataset(encoded[:1_000_000], length = length, shuffle = True, seed=42)
valid_set = to_dataset(encoded[1_000_000:1_060_000], length = length)
train_set = to_dataset(encoded[1_060_000:], length = length)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim = n_tokens, output_dim = 16),
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.Dense(n_tokens, activation="softmax")
])    

model.compile(loss="sparse_categorical_crossentropy", optimizer = "nadam", metrics=["accuracy"])
model_ckpt = tf.keras.callbacks.ModelCheckpoint("my_shakespeare_model.keras", monitor="val_accuracy", save_best_only=True)


history = model.fit(train_set, validation_data = valid_set, epochs=1, callbacks=[model_ckpt])

shakespeare_model = tf.keras.Sequential([
    text_vec_layer,
    tf.keras.layers.Lambda(lambda X:X-2), model
])

def next_char(text, temperature=1):
    # Ensure text is a list of strings (sequence of characters)
    text = [text[0]]
    y_proba = shakespeare_model.predict(text)[0,-1:]

    rescaled_logits = tf.math.log(y_proba)/temperature 
    char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0,0]
    return text_vec_layer.get_vocabulary()[char_id+2]


def extend_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text_as_list = [text]
        next_character = next_char(text_as_list, temperature)
        text += next_character
    return text