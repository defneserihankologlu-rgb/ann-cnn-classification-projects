

## ✅ ** README.md**

````markdown
# ANN & CNN Classification Projects (TensorFlow / Keras)

This repository contains two supervised learning projects implemented with
TensorFlow/Keras, covering both **tabular data (ANN)** and **image data (CNN)**
classification pipelines.

---

## Repository Structure

```text
.
├── braille_cnn/
│   └── train.py
├── fetal_health_ann/
│   └── train.py
├── README.md
````

---

## 1) Fetal Health Classification (ANN)

**Task:** 3-class classification of fetal health status
**Classes:** Normal / Suspect / Pathological

* **Dataset:** Kaggle — `andrewmvd/fetal-health-classification`
* **Data type:** Tabular (21 numerical features extracted from CTG exams)
* **Model:** Multilayer Perceptron (ANN)

⚠️ **Important note:**
The dataset is **class-imbalanced** (Normal class dominates).
Therefore, accuracy alone may be misleading — class-wise precision, recall,
and F1-score should be considered, especially for the *Pathological* class.

### Run

```bash
pip install -r requirements.txt
python fetal_health_ann/train.py
```

### Outputs

* Printed **classification report**
* Printed **confusion matrix**
* (Optional) training accuracy / loss curves if enabled in the script

---

## 2) Braille Character Recognition (CNN)

**Task:** 26-class image classification (A–Z)
**Image size:** 28×28 grayscale

* **Dataset:** Kaggle — `shanks0465/braille-character-dataset`
* **Model:** Convolutional Neural Network (CNN)

The dataset includes multiple augmented versions of each character
(rotation, shift, brightness), improving robustness.

### Run

```bash
pip install -r requirements.txt
python braille_cnn/train.py
```

### Outputs

After training, the following files are generated in the repository root:

* `confusion_matrix.png`
* `training_history.png`

Additionally, a **classification report** is printed to the console.

---

## Datasets

Datasets are **not included** in this repository.

They are downloaded programmatically using `kagglehub` inside the scripts:

* Fetal Health: `andrewmvd/fetal-health-classification`
* Braille: `shanks0465/braille-character-dataset`

Make sure your Kaggle environment is properly configured if required.

---

## Requirements

Create a `requirements.txt` file with:

```txt
kagglehub
numpy
pandas
scikit-learn
matplotlib
Pillow
tensorflow
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Notes

* ANN is preferred for tabular CTG features.
* CNN is used to exploit spatial patterns in Braille images.
* EarlyStopping, class weighting, and data augmentation can further improve results.

---

## License

This project is intended for educational and academic use.

Hazırsan “README’yi ekledim” de, son kontrolü yapalım 👌
```
