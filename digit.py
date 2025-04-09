import numpy as np
import os
from PIL import Image, ImageDraw
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import tkinter as tk


img_w, img_h = 28, 28
model_path = "digit_cnn_model.h5"

def train_and_save_model():
    print("🔁 Training model for the first time...")

    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess
    X_train = X_train.reshape(-1, img_w, img_h, 1).astype('float32') / 255
    X_test = X_test.reshape(-1, img_w, img_h, 1).astype('float32') / 255
    y_train_cat = keras.utils.to_categorical(y_train)
    y_test_cat = keras.utils.to_categorical(y_test)

    # Build model
    model = keras.models.Sequential([
        layers.Conv2D(32, (3, 3), input_shape=(img_w, img_h, 1)),
        layers.LeakyReLU(alpha=0.3),
        layers.BatchNormalization(),

        layers.Conv2D(32, (3, 3)),
        layers.LeakyReLU(alpha=0.3),
        layers.BatchNormalization(),

        layers.Conv2D(32, (3, 3)),
        layers.LeakyReLU(alpha=0.3),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3)),
        layers.LeakyReLU(alpha=0.3),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128),
        layers.LeakyReLU(alpha=0.3),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train_cat, epochs=10, batch_size=128, verbose=1, validation_data=(X_test, y_test_cat))
    model.save(model_path)
    print("✅ Model trained and saved.")
    return model

# Load or train model
if os.path.exists(model_path):
    print("✅ Loading saved model...")
    model = load_model(model_path)
else:
    model = train_and_save_model()

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Digit Recognizer")
        self.canvas = tk.Canvas(self, width=280, height=280, bg="white")
        self.canvas.pack()
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)

        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)

        btn_frame = tk.Frame(self)
        btn_frame.pack()

        tk.Button(btn_frame, text="Predict", command=self.predict_digit).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Clear", command=self.clear_canvas).pack(side=tk.LEFT)
        self.result_label = tk.Label(self, text="Draw a digit!", font=("Arial", 16))
        self.result_label.pack()

    def paint(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill="white")
        self.result_label.config(text="Draw a digit!")

    def predict_digit(self):
        # Resize and preprocess the image
        img = self.image.resize((28, 28)).convert("L")
        img_array = np.array(img).reshape(1, 28, 28, 1).astype("float32") / 255

        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        self.result_label.config(text=f"Prediction: {digit} ({confidence:.2f}%)")

# Run the app
if __name__ == "__main__":
    app = App()
    app.mainloop()
