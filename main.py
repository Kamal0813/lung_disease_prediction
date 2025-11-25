import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Input, Concatenate,
    Reshape, GlobalMaxPooling1D, GlobalAveragePooling1D,
    TimeDistributed, MultiHeadAttention, LayerNormalization,
    Lambda, Embedding, Flatten
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# ============================
# 1. LOAD & BASIC PREPROCESS
# ============================
df = pd.read_csv("lung_cancer.csv")
print("Dataset shape:", df.shape)
print(df.head())

# Encode GENDER (M/F -> 0/1)
le_gender = LabelEncoder()
df["GENDER"] = le_gender.fit_transform(df["GENDER"])

# Encode TARGET (LUNG_CANCER: YES/NO -> 1/0)
TARGET_COL = "LUNG_CANCER"
le_target = LabelEncoder()
df[TARGET_COL] = le_target.fit_transform(df[TARGET_COL])

# Convert all 1/2 binary columns to 0/1 (except already handled)
binary_cols = [
    col for col in df.columns
    if df[col].nunique() == 2 and col not in ["GENDER", TARGET_COL]
]
for col in binary_cols:
    if set(df[col].unique()) == {1, 2}:
        df[col] = df[col] - 1

# ============================
# 2. FEATURE ENGINEERING
# ============================
# Add interaction features (binary × binary)
df["SMOKING_YELLOW_FINGERS"] = df["SMOKING"] * df["YELLOW_FINGERS"]
df["SMOKING_ANXIETY"] = df["SMOKING"] * df["ANXIETY"]
df["ALCOHOL_FATIGUE"] = df["ALCOHOL_CONSUMING"] * df["FATIGUE"]
df["COUGHING_CHEST_PAIN"] = df["COUGHING"] * df["CHEST_PAIN"]
df["SHORT_BREATH_CHEST_PAIN"] = df["SHORTNESS_OF_BREATH"] * df["CHEST_PAIN"]

# Add age-based interaction features (continuous)
df["AGE_SMOKING"] = df["AGE"] * df["SMOKING"]
df["AGE_ALCOHOL"] = df["AGE"] * df["ALCOHOL_CONSUMING"]
df["AGE_CHEST_PAIN"] = df["AGE"] * df["CHEST_PAIN"]

print("\nColumns after feature engineering:", df.columns.tolist())

# ============================
# 3. FEATURES & TARGET
# ============================
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].values  # 0/1

num_classes = 2

# ============================
# 4. TRAIN–TEST SPLIT
# ============================
X_train_df, X_test_df, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# 5. SCALE NUMERICAL COLUMNS
# ============================
numeric_cols = ["AGE", "AGE_SMOKING", "AGE_ALCOHOL", "AGE_CHEST_PAIN"]
other_cols = [c for c in X.columns if c not in numeric_cols]

scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train_df[numeric_cols])
X_test_num = scaler.transform(X_test_df[numeric_cols])

X_train_other = X_train_df[other_cols].values.astype("float32")
X_test_other = X_test_df[other_cols].values.astype("float32")

# final input: [scaled numeric features] + other binary/engineered features
X_train = np.concatenate([X_train_num, X_train_other], axis=1)
X_test = np.concatenate([X_test_num, X_test_other], axis=1)

input_dim = X_train.shape[1]
print("Input dimension:", input_dim)

# One-hot encode labels
y_train = to_categorical(y_train_raw, num_classes=num_classes)
y_test = to_categorical(y_test_raw, num_classes=num_classes)

# ============================
# 6. CLASS WEIGHTS
# ============================
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_raw),
    y=y_train_raw
)
class_weights = {i: w for i, w in enumerate(class_weights_array)}
print("Class weights:", class_weights)

# ============================
# 7. CALLBACKS
# ============================
CALLBACKS = [
    # EarlyStopping(
    #     monitor="val_loss",
    #     patience=15,
    #     restore_best_weights=True
    # ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=7,
        min_lr=1e-5
    )
]

# ============================
# 8. MODEL DEFINITIONS
# ============================

# 8.1 ANN (with engineered features)
def build_ann():
    model = Sequential([
        Dense(128, activation="relu", input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.1),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# 8.2 Deep Neural Network (DNN)
def build_dnn():
    model = Sequential([
        Dense(256, activation="relu", input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# 8.3 Dropout + BatchNorm Deep Network
def build_dropout_bn():
    model = Sequential([
        Dense(512, activation="relu", input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.35),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.35),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# 8.4 Deep & Cross Network (DCN)
def cross_layer(x0, x, num_layers=3):
    for _ in range(num_layers):
        dot = Dense(1)(x)        # (batch,1)
        cross = x0 * dot         # broadcast
        x = cross + x            # residual-like
    return x

def build_dcn():
    inputs = Input(shape=(input_dim,))
    deep = Dense(128, activation="relu")(inputs)
    deep = Dense(64, activation="relu")(deep)

    cross = cross_layer(inputs, inputs)

    combined = Concatenate()([deep, cross])
    x = Dense(64, activation="relu")(combined)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# 8.5 Transformer-style model for tabular data
def build_transformer_tabular(d_model=64, num_heads=4, ff_dim=128):
    inputs = Input(shape=(input_dim,))

    x = Reshape((input_dim, 1))(inputs)
    x = TimeDistributed(Dense(d_model))(x)

    attn = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads
    )(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attn)

    ff = Sequential([
        Dense(ff_dim, activation="relu"),
        Dense(d_model)
    ])
    x = LayerNormalization(epsilon=1e-6)(x + ff(x))

    x = GlobalAveragePooling1D()(x)

    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# 8.6 Autoencoder + Classifier (uses engineered features + class_weight)
def train_autoencoder_classifier(X_train, X_test, y_train, y_test):
    encoding_dim = 32

    inp = Input(shape=(input_dim,))
    encoded = Dense(128, activation="relu")(inp)
    encoded = Dense(encoding_dim, activation="relu")(encoded)
    decoded = Dense(128, activation="relu")(encoded)
    decoded = Dense(input_dim, activation="linear")(decoded)

    autoencoder = Model(inp, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")

    autoencoder.fit(
        X_train, X_train,
        epochs=200,
        batch_size=16,
        validation_split=0.2,
        verbose=0,
        callbacks=CALLBACKS
    )

    encoder = Model(inp, encoded)
    X_train_enc = encoder.predict(X_train)
    X_test_enc = encoder.predict(X_test)

    clf_inp = Input(shape=(encoding_dim,))
    x = Dense(64, activation="relu")(clf_inp)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)
    clf = Model(clf_inp, out)
    clf.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    clf.fit(
        X_train_enc, y_train,
        epochs=200,
        batch_size=16,
        validation_split=0.2,
        verbose=0,
        callbacks=CALLBACKS,
        class_weight=class_weights
    )

    probs = clf.predict(X_test_enc, verbose=0)
    y_pred = probs.argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)

    return acc, y_pred

# ============================
# 9. TRAIN ALL MODELS
# ============================
EPOCHS = 20
BATCH_SIZE = 64

models = {
    "ANN": build_ann(),
    "DNN": build_dnn(),
    "Dropout_BN": build_dropout_bn(),
    "DCN": build_dcn(),
    "Transformer_Tabular": build_transformer_tabular()
}

results = {}
y_pred_dict = {}

print("\n================ TRAINING DEEP LEARNING MODELS (WITH FEATURE ENGINEERING) ================\n")

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=0,
        callbacks=CALLBACKS,
        class_weight=class_weights
    )
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    results[name] = acc

    probs = model.predict(X_test, verbose=0)
    y_pred = probs.argmax(axis=1)
    y_pred_dict[name] = y_pred

    print(f"{name} Accuracy: {acc:.4f}\n")

print("Training Autoencoder_Classifier...")
ae_acc, ae_pred = train_autoencoder_classifier(X_train, X_test, y_train, y_test)
results["Autoencoder_Classifier"] = ae_acc
y_pred_dict["Autoencoder_Classifier"] = ae_pred
print(f"Autoencoder_Classifier Accuracy: {ae_acc:.4f}\n")

# ============================
# 10. ACCURACY TABLE & CSV
# ============================
results_df = pd.DataFrame([
    {"Model": name, "Accuracy": acc}
    for name, acc in results.items()
])

print("\n================ MODEL ACCURACY TABLE (WITH FEATURE ENGINEERING) ================\n")
print(results_df)

results_df.to_csv("model_accuracies_feature_engineered.csv", index=False)
print("\nModel accuracies saved to 'model_accuracies_feature_engineered.csv'")

# ============================
# 11. CONFUSION MATRIX (BEST MODEL)
# ============================
best_model_name = max(results, key=results.get)
best_acc = results[best_model_name]

print(f"\nBest Model: {best_model_name} with accuracy {best_acc:.4f}")

best_y_pred = y_pred_dict[best_model_name]
best_y_true = y_test_raw

cm = confusion_matrix(best_y_true, best_y_pred)
print("\nConfusion Matrix (Best Model):")
print(cm)

print("\nClassification Report (Best Model):")
print(classification_report(best_y_true, best_y_pred, digits=4))
