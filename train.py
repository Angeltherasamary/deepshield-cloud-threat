import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D, Bidirectional, GRU, Dropout, Dense
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import generate_telemetry_dataset

def build_model(input_shape=(10, 20), num_classes=4):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Bidirectional(GRU(64, return_sequences=False)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    X, y = generate_telemetry_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = build_model()
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

    model.save("cloudshield_model.keras")
    print("Saved weights to cloudshield_model.keras")

    # Evaluation & Plots
    # 1. Accuracy/Loss Curve
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history['accuracy'], label='Train')
    ax[0].plot(history.history['val_accuracy'], label='Val')
    ax[0].set_title('Accuracy Curve')
    ax[0].legend()

    ax[1].plot(history.history['loss'], label='Train')
    ax[1].plot(history.history['val_loss'], label='Val')
    ax[1].set_title('Loss Curve')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("viz_acc_loss_curve.png", dpi=300)
    plt.close()

    # 2. Confusion Matrix
    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    labels = ['Benign', 'DDoS Flood', 'PortScan', 'BruteForce']
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig("viz_confusion_matrix.png", dpi=300)
    plt.close()

    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=labels))

if __name__ == "__main__":
    main()