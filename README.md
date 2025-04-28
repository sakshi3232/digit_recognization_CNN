# 🧠 Handwritten Digit Recognizer

This is a simple Python-based GUI application that allows users to draw a digit on screen and get it recognized using a Convolutional Neural Network (CNN) trained on the MNIST dataset.


## 📦 Files in the Repository

- `digit.py` – Main application file containing the model training logic and GUI interface using Tkinter.
- `digit_cnn_model.h5` – Pretrained CNN model for recognizing digits.
- `requirements.txt` – Dependencies needed to run the application.

---

## 🚀 How It Works

1. **Model Training**
   - If a pretrained model (`digit_cnn_model.h5`) is not found, the script automatically trains a CNN on the MNIST dataset.
   - The model is saved for future use, so training only happens once.

2. **GUI Drawing Interface**
   - A Tkinter canvas lets users draw digits using their mouse.
   - The drawing is resized to 28x28 pixels (MNIST format), preprocessed, and passed to the model.
   - The predicted digit and confidence score are displayed instantly.

---

## 🧠 Model Architecture

- 3 Convolutional Layers (LeakyReLU activations)
- Batch Normalization & Dropout for regularization
- MaxPooling for downsampling
- Dense layer before the output
- Softmax output layer for classification into 10 digits (0–9)

---

## 🖥️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Install dependencies

Make sure you have Python 3 and pip installed. Then:

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
python digit.py
```

---

## ✅ Dependencies

List of libraries used:

- `tensorflow`
- `numpy`
- `Pillow`
- `tkinter` 

---

## ✨ Features

- Real-time digit recognition
- No need to retrain every time
- Clean and intuitive interface
- Confidence score display
- Clear canvas button


