# Image Captioning using InceptionV3 and LSTM

## Overview
This project implements an image captioning model using InceptionV3 for image feature extraction and an LSTM-based sequence model for generating captions. The COCO 2017 training dataset is used.

## Prerequisites
Install required libraries:
```bash
pip install tensorflow numpy matplotlib scikit-learn pycocotools
```

## Dataset Preparation
- Dataset: COCO 2017
- Annotations file: `captions_train2017.json`
- Image directory: `train2017`

## Loading COCO Annotations
```python
coco = COCO(annFile)
imgIds = coco.getImgIds()
```
Extracts image IDs and captions.

## Caption Preprocessing
Tokenization and padding ensure uniform input length:
```python
def preprocess_captions(captions, num_words=5000, max_length=34):
    tokenizer = Tokenizer(num_words=num_words, oov_token="<unk>")
    tokenizer.fit_on_texts(captions)
    sequences = tokenizer.texts_to_sequences(captions)
    return tokenizer, pad_sequences(sequences, padding='post', maxlen=max_length), tokenizer.word_index
```

## Image Feature Extraction
InceptionV3 extracts feature vectors:
```python
base_model = InceptionV3(weights="imagenet")
feature_extractor = tf.keras.Model(base_model.input, base_model.layers[-2].output)
```

## Splitting Data
Training and validation sets are created:
```python
X_image_train, X_image_val, y_caption_train, y_caption_val = train_test_split(
    image_features, padded_captions, test_size=0.2, random_state=42
)
```

## Target Caption Preparation
Teacher forcing splits captions into input-output sequences:
```python
def create_target_captions(captions, sequence_length):
    return pad_sequences(captions[:, :-1], padding='post', maxlen=sequence_length), to_categorical(captions[:, 1:], num_classes=vocab_size)
```

## Model Architecture
A dual-branch model integrates image and text inputs:
```python
image_input = Input(shape=(2048,))
image_dense = Dense(embedding_dim, activation="relu")(image_input)
image_repeated = RepeatVector(sequence_length)(image_dense)
text_input = Input(shape=(sequence_length,))
text_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(text_input)
text_lstm = LSTM(256, return_sequences=True)(text_embedding)
combined = Add()([image_repeated, text_lstm])
output = TimeDistributed(Dense(vocab_size, activation="softmax"))(combined)
model = Model(inputs=[image_input, text_input], outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy")
```

## Training
```python
history = model.fit(
    [X_image_train, y_caption_train_input],
    y_caption_train_output,
    validation_data=([X_image_val, y_caption_val_input], y_caption_val_output),
    batch_size=64,
    epochs=5,
)
```

## Saving Model & Tokenizer
```python
model.save("coco_image_caption_generator_model.h5")
with open("coco_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
```

## Training Loss Visualization
```python
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.savefig("coco_training_history.png")
plt.show()
```

## Conclusion
This project integrates CNNs and LSTMs for image captioning. Future improvements can include Transformer-based architectures.

