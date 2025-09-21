# MNIST One-vs-All Digit Classifier

This project trains 10 separate binary classifiers (one for each digit 0–9) on the MNIST dataset using a One-vs-All (OvR) approach:

- Model 0: predicts "is this digit 0?"
- Model 1: predicts "is this digit 1?"
- ...
- Model 9: predicts "is this digit 9?"

For each input image, all 10 models are evaluated, and the digit with the highest probability is selected as the predicted class.

---

## Features

- Train individual binary classifiers for each digit.
- Multiclass prediction using argmax across all 10 models.
- Predict your own handwritten digits (PNG/JPG) in my_digits/.
- Supports large datasets via a CSV file (data_labels.csv).
- Uses PyTorch + torchvision for training and evaluation.
- Automatic preprocessing: grayscale, resize to 28x28, normalization.

---

## Project Structure

MNIST_Model/
- mnist_ovr.py          # Main script (train, evaluate, predict)
- saved_models/         # Trained models (digit_0.pth … digit_9.pth)
- data/                 # MNIST dataset (auto-downloaded)
- my_digits/            # Folder for your handwritten images
- data_labels.csv       # Optional CSV for large datasets
- README.md             # Project documentation

---

## Installation

Clone the repository:

git clone https://github.com/PreranRai/MNIST_Model.git
cd MNIST_Model

Install dependencies:

pip install torch torchvision matplotlib numpy pillow pandas

---

## Training

Run the main script:

python mnist_ovr.py

- If saved_models/ is empty, the script trains all 10 binary classifiers and saves them.
- Adjust epochs or batch size inside train_and_save_models() for better accuracy.

---

## Evaluation

- The script automatically evaluates the MNIST test set.
- Displays sample MNIST images with predicted and true labels.

---

## Predict Your Own Handwritten Digits

1. Place your images in the my_digits/ folder (PNG or JPG).
2. Run:

python mnist_ovr.py

Example output:

img1.png → Predicted digit: 7
img2.jpg → Predicted digit: 3
...

---

## Predict from a CSV Dataset (Optional)

- Create data_labels.csv with format:

path,label
my_dataset/img001.png,5
my_dataset/img002.png,3
...

- The script will predict all images in the CSV automatically.

---

## Notes

- Images must be single digits, preferably black on white.
- Preprocessing automatically resizes images to 28x28 pixels.
- CPU-only PyTorch works fine; GPU optional for faster training.

---

## Future Improvements

- Train a single multiclass classifier (10 outputs) for better performance.
- Add EMNIST support (letters + digits).
- Enhance preprocessing (deskewing, centering, noise removal).

---

## License

This project is licensed under the MIT License.
