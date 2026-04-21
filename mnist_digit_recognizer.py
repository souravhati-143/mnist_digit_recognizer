# ============================================================
# Project 2: Handwritten Digit Recognizer (MNIST + CNN)
# Author: Sourav Hati | Enrollment: ISL-869225
# Internship: Codec Technologies - Python Developer Internship
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("  HANDWRITTEN DIGIT RECOGNIZER - CNN on MNIST")
print("  By: Sourav Hati | Codec Technologies 2026")
print("=" * 55)

# ─────────────────────────────────────────
# STEP 1: Try TensorFlow, fallback to sklearn
# ─────────────────────────────────────────
try:
    import tensorflow as tf
    keras = tf.keras
    layers = tf.keras.layers
    USE_CNN = True
    print("\n✅ TensorFlow found — using CNN model")
except ImportError:
    from sklearn.datasets import fetch_openml
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.model_selection import train_test_split
    USE_CNN = False
    print("\n⚠️  TensorFlow not found — using Random Forest (still valid!)")

# ─────────────────────────────────────────
# STEP 2: Load MNIST Dataset
# ─────────────────────────────────────────
print("\n📥 Loading MNIST Dataset...")

if USE_CNN:
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_test  = X_test.astype('float32') / 255.0
    X_train_cnn = X_train[..., np.newaxis]
    X_test_cnn  = X_test[..., np.newaxis]
    print(f"✅ Train: {X_train.shape} | Test: {X_test.shape}")
else:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data / 255.0, mnist.target.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✅ Train: {X_train.shape} | Test: {X_test.shape}")

# ─────────────────────────────────────────
# STEP 3: Build & Train Model
# ─────────────────────────────────────────
if USE_CNN:
    print("\n🏗️  Building CNN Model...")
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    print("\n🚀 Training CNN (5 epochs)...")
    history = model.fit(X_train_cnn, y_train, epochs=5, batch_size=64,
                        validation_split=0.1, verbose=1)

    test_loss, test_acc = model.evaluate(X_test_cnn, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test_cnn), axis=1)
    print(f"\n🎯 CNN Test Accuracy: {test_acc*100:.2f}%")

else:
    print("\n🚀 Training Random Forest (n=100)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train[:10000], y_train[:10000])  # subset for speed
    y_pred = model.predict(X_test[:2000])
    y_test_eval = y_test[:2000]
    test_acc = accuracy_score(y_test_eval, y_pred)
    print(f"\n🎯 Random Forest Accuracy: {test_acc*100:.2f}%")
    y_pred_full = model.predict(X_test)

# ─────────────────────────────────────────
# STEP 4: Visualize Results
# ─────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Handwritten Digit Recognizer - MNIST\nSourav Hati | Codec Technologies 2026',
             fontsize=14, fontweight='bold')

# Sample predictions grid
gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.4)

# Top: Sample digit predictions
ax_top = fig.add_subplot(gs[0])
ax_top.axis('off')
ax_top.set_title('Sample Predictions (20 random test images)', fontweight='bold', pad=10)

if USE_CNN:
    sample_images = X_test[:20]
    sample_true = y_test[:20]
    sample_pred = y_pred[:20]
else:
    imgs_2d = X_test[:20].reshape(-1, 28, 28)
    sample_images = imgs_2d
    sample_true = y_test[:20]
    sample_pred = y_pred_full[:20]

inner_gs = gridspec.GridSpecFromSubplotSpec(2, 10, subplot_spec=gs[0], hspace=0.5, wspace=0.3)
for i in range(20):
    ax = fig.add_subplot(inner_gs[i//10, i%10])
    img = sample_images[i].reshape(28,28)
    ax.imshow(img, cmap='gray')
    true_label = int(sample_true[i])
    pred_label = int(sample_pred[i])
    color = 'green' if true_label == pred_label else 'red'
    ax.set_title(f'P:{pred_label}', fontsize=8, color=color, pad=2)
    ax.axis('off')

# Middle: Confusion matrix heatmap (simple)
ax_mid = fig.add_subplot(gs[1])
if USE_CNN:
    cm_pred = y_pred
    cm_true = y_test
else:
    cm_pred = y_pred_full
    cm_true = y_test

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(cm_true[:2000], cm_pred[:2000])
im = ax_mid.imshow(cm, cmap='Blues', interpolation='nearest')
ax_mid.set_title('Confusion Matrix (first 2000 test samples)', fontweight='bold')
ax_mid.set_xlabel('Predicted Label')
ax_mid.set_ylabel('True Label')
ax_mid.set_xticks(range(10))
ax_mid.set_yticks(range(10))
for i in range(10):
    for j in range(10):
        ax_mid.text(j, i, str(cm[i,j]), ha='center', va='center',
                   fontsize=7, color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=ax_mid)

# Bottom: Per-digit accuracy bar chart
ax_bot = fig.add_subplot(gs[2])
digit_acc = []
for digit in range(10):
    mask = (np.array(cm_true[:2000]) == digit)
    if mask.sum() > 0:
        acc = (np.array(cm_pred[:2000])[mask] == digit).mean() * 100
        digit_acc.append(acc)
    else:
        digit_acc.append(0)

colors = ['#2ecc71' if a >= 95 else '#e67e22' if a >= 90 else '#e74c3c' for a in digit_acc]
bars = ax_bot.bar(range(10), digit_acc, color=colors, edgecolor='white', linewidth=0.5)
ax_bot.set_title('Per-Digit Recognition Accuracy', fontweight='bold')
ax_bot.set_xlabel('Digit')
ax_bot.set_ylabel('Accuracy (%)')
ax_bot.set_xticks(range(10))
ax_bot.set_ylim(80, 101)
ax_bot.axhline(y=95, color='gray', linestyle='--', alpha=0.5, label='95% line')
ax_bot.legend()
for bar, acc in zip(bars, digit_acc):
    ax_bot.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
               f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)

plt.savefig('mnist_output.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Chart saved as 'mnist_output.png'")
print(f"\n📊 FINAL RESULT: {'CNN' if USE_CNN else 'Random Forest'} Accuracy = {test_acc*100:.2f}%")
print("\n🎉 Project 2 Complete! Ready to upload to GitHub.")