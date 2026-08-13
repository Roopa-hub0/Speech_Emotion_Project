# VocalSense AI Real-Time Speech-Based Emotion, Gender and Stress Monitoring System Using Deep Learning

VocalSense AI is a real-time speech analysis system that combines **Deep Learning and Machine Learning** to recognize emotions, predict speaker gender, and estimate stress levels from speech. The system uses an **Attention-based CNN** for emotion recognition and a **Random Forest classifier** with MFCC features for gender prediction.

## Features

* 🎙️ Real-time speech recording
* 📁 Audio file upload
* 😊 Recognition of 8 emotions
* 👤 Speaker gender prediction
* 🧠 Attention CNN-based emotion classification
* 🌲 Random Forest-based gender classification
* 😌 Rule-based stress estimation
* 📊 Prediction history and analytics
* 🌐 Flask-based web application

## Emotions Recognized

The system classifies speech into eight emotion categories:

* Angry
* Calm
* Disgust
* Fearful
* Happy
* Neutral
* Sad
* Surprised

## System Architecture

The speech input is processed through two parallel branches:

```text
                Speech Input
                     │
          ┌──────────┴──────────┐
          │                     │
     Preprocessing         Preprocessing
          │                     │
   Mel Spectrogram             MFCC
          │                     │
    Attention CNN         Random Forest
          │                     │
     Emotion                Gender
          │
   Rule-Based Mapping
          │
        Stress
          │
   ┌──────┴───────┐
   │ Web Interface│
   └──────────────┘
```

## Dataset

The project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset for speech emotion recognition.

The dataset is processed to generate:

* Mel Spectrogram images for the Attention CNN
* MFCC feature vectors for gender classification

## Models

### 1. Attention CNN

The emotion recognition model uses:

* Input size: `128 × 128 × 3`
* Convolutional layers with 32, 64 and 128 filters
* Kernel size: `3 × 3`
* ReLU activation
* Max pooling
* Channel-based SE (Squeeze-and-Excitation) attention mechanism
* Dense layers with 256 and 128 neurons
* Dropout
* Softmax output layer for 8 emotion classes

### 2. Random Forest

MFCC features are used as input to a Random Forest classifier for speaker gender prediction.

### 3. Stress Estimation

Stress is estimated using a rule-based mapping from the detected emotion:

| Emotion   | Estimated Stress |
| --------- | ---------------- |
| Happy     | Low              |
| Calm      | Low              |
| Neutral   | Medium           |
| Surprised | Medium           |
| Sad       | Medium-High      |
| Angry     | High             |
| Fearful   | High             |
| Disgust   | High             |

> Stress estimation is a heuristic component and should not be considered a clinical diagnosis.

## Experimental Configuration

| Parameter                | Value                           |
| ------------------------ | ------------------------------- |
| Programming Language     | Python                          |
| Deep Learning Framework  | TensorFlow / Keras              |
| Machine Learning Library | Scikit-learn                    |
| Audio Processing         | Librosa                         |
| Dataset                  | RAVDESS                         |
| Train-Validation Split   | 80:20                           |
| Image Size               | 128 × 128                       |
| Batch Size               | 32                              |
| Epochs                   | 25                              |
| Optimizer                | Adam                            |
| Learning Rate            | 0.001                           |
| Loss Function            | Sparse Categorical Crossentropy |
| Early Stopping           | Patience = 5                    |

## Performance

The reported experimental results are:

* **Emotion Recognition Accuracy:** 88.72%
* **Gender Classification Accuracy:** 97.57%

The system also provides real-time prediction through a Flask web application.

## Technologies Used

* Python
* TensorFlow
* Keras
* Scikit-learn
* Librosa
* NumPy
* Pandas
* Matplotlib
* Flask
* Joblib

## Project Structure

```text
VocalSense-AI/
│
├── data/
│   └── RAVDESS/
│
├── models/
│   ├── attention_cnn_model.h5
│   └── emotion_model.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── train_attention_cnn.py
│   ├── train_random_forest.py
│   └── prediction.py
│
├── templates/
│   └── index.html
│
├── static/
│   └── ...
│
├── augmented_spectrogram_dataset/
│
├── requirements.txt
├── app.py
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/VocalSense-AI.git
cd VocalSense-AI
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Flask application:

```bash
python app.py
```

Then open the local address shown by Flask in your browser.

You can either:

1. Record speech using the microphone, or
2. Upload an audio file.

The system processes the speech and displays the predicted emotion, gender, estimated stress level, and related analytics.

## Reproducibility

The dataset, model architecture, preprocessing procedure, training configuration, and evaluation methodology are described in this repository. Additional implementation details and experimental materials can be provided upon reasonable request.

## Ethical Considerations

VocalSense AI is intended for research and assistive purposes only. Emotion and stress predictions should not be considered clinical diagnoses or the sole basis for decision-making. Speech data should be collected with informed consent and handled according to applicable privacy regulations.

## Disclaimer

The stress estimation component is based on predefined emotion-to-stress mappings and is **not a medical or psychological diagnostic tool**.

## Authors

Roopa Ramthapure and collaborators

Developed as a research project on **speech emotion recognition, gender prediction, and stress monitoring using AI/ML**.
