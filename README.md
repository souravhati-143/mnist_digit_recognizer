# 🔢 Handwritten Digit Recognizer (MNIST)

**Internship Project | Codec Technologies Python Developer Internship 2026**
**Author:** Sourav Hati | Rajdhani College, Bhubaneswar

---

## 📌 Project Overview
A Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize handwritten digits (0–9) with high accuracy. Demonstrates deep learning, image processing, and model evaluation skills.

## 🛠️ Technologies Used
- Python 3.x
- TensorFlow / Keras (CNN model)
- NumPy & Matplotlib
- Scikit-learn (evaluation metrics)

## 🧠 Model Architecture (CNN)
```
Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128) → Dropout → Dense(10)
```

## 📊 Features
- ✅ CNN trained on 60,000 MNIST images
- ✅ ~99% test accuracy
- ✅ Sample prediction visualization (20 images)
- ✅ Confusion matrix heatmap
- ✅ Per-digit accuracy bar chart
- ✅ Falls back to Random Forest if TensorFlow not installed

## 🚀 How to Run

```bash
# Install dependencies
pip install tensorflow numpy matplotlib scikit-learn

# Run the project
python mnist_digit_recognizer.py
```

## 📈 Output
- Console: Training progress + final accuracy
- Chart: `mnist_output.png` with predictions, confusion matrix, per-digit accuracy

## 📬 Submitted to
Codec Technologies | vaishali@codectechnologies.in
