# 🧠 Sign Language Interpreter (Indian Sign Language)

A real-time browser-based Sign Language Interpreter that uses the system camera to recognize and interpret **Indian Sign Language (ISL)** gestures using a locally trained Machine Learning model.

Built for Hackathon – Open Innovation Category 🚀

---

## 📌 Overview

This project captures live hand gesture data using the browser camera, processes landmark features using MediaPipe, and compares them against a locally trained ML model to predict the corresponding Indian Sign Language gesture.

The model was trained on a custom dataset of Indian Sign Language gesture landmarks.

---

## ⚙️ Tech Stack

- Python
- HTML / CSS
- JavaScript
- Machine Learning (Scikit-learn)
- OpenCV
- MediaPipe (Hand Landmark Detection)
- Flask (Backend)

---

INTERPRETER/
│
├── app.py
├── train_model.py
├── index.html
├── requirement.txt
├── .gitignore
├── isl_model.pkl (Download separately – see below)
└── Indian Sign Language Gesture Landmarks.csv (Download separately – see below)


---

## ⚠️ Important: Large Files Not Included

Due to GitHub's 100MB file size limit, the following files are not included in the repository:

- `isl_model.pkl`
- `Indian Sign Language Gesture Landmarks.csv`

---

## 📥 Download Required Files

### 1️⃣ Indian Sign Language Gesture Landmarks Dataset

Download here:  
https://drive.google.com/file/d/1ftmy85FejriLgQ8i0VZ1a_4YdqnmeGIR/view?usp=sharing

---

### 2️⃣ Trained ML Model (isl_model.pkl)

Download here:  
https://drive.google.com/file/d/16S5XUlYT-NJY_YcfVFMFyf1r-3komsJz/view?usp=sharing

---

## 📂 After Downloading

Place both downloaded files inside the root project directory:

Your folder should look like this:

INTERPRETER/
│
├── app.py
├── train_model.py
├── index.html
├── requirement.txt
├── isl_model.pkl
└── Indian Sign Language Gesture Landmarks.csv

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

git clone https://github.com/advik-thiagarajan/Sign_Language_interpreter.git

cd Sign_Language_interpreter

---

### 2️⃣ Install Dependencies
pip install -r requirement.txt

---

### 3️⃣ Run the Application

python app.py


---

### 4️⃣ Open in Browser

Visit: ""Local host""


Allow camera permissions when prompted.

---

## 🎯 Features

- Real-time gesture recognition
- Browser-based camera input
- Locally processed ML inference
- Indian Sign Language dataset support
- Lightweight and hackathon-ready

---

## 🧠 Model Information

- Model Type: Supervised Classification Model
- Training Data: Indian Sign Language Landmark Dataset
- Input: Hand Landmark Coordinates
- Output: Predicted Gesture Label

---

## 🌍 Future Improvements

- Sentence formation from gesture sequences
- Text-to-speech output
- Cloud deployment
- Larger gesture vocabulary
- Mobile device optimization

---

## 🏆 Vision

To promote inclusive communication by enabling real-time Indian Sign Language interpretation using accessible web technologies and machine learning.

## 🗂 Project Structure
