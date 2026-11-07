# Real-Time Drowsiness Detection System

A Python-based real-time drowsiness detection system that uses computer
vision and facial landmarks to monitor driver behavior and identify
signs of fatigue.

The system analyzes eye movements, yawning, and head movements through a
webcam to help detect possible drowsiness.

## Features

-   **Eye Blink Detection:** Detects and counts eye blinks in real time.
-   **Sleepy Eye Detection:** Monitors eye closure using the Eye Aspect
    Ratio (EAR).
-   **Yawning Detection:** Identifies yawning using the Mouth Aspect
    Ratio (MAR).
-   **Head Nod Detection:** Detects head movements that may indicate
    drowsiness.
-   **Real-Time Monitoring:** Processes webcam video continuously.
-   **Alarm Support:** Uses Pygame for audio alarm functionality.
-   **Event Counters:** Tracks blinks, yawns, sleepy-eye events, and
    head nods.
-   **Session Summary:** Displays detection statistics when the
    application exits.

## Technologies Used

-   Python
-   OpenCV
-   dlib
-   NumPy
-   SciPy
-   Pygame
-   Facial Landmark Detection (68-point model)

## Project Structure

``` text
Drowsiness-Detection-System/
├── drowsiness_model/
│   ├── alarm.wav
│   ├── config.json
│   ├── metadata.json
│   └── model.weights.h5
├── shape_predictor_68_face_landmarks.dat
├── webcam_drowsiness.py
└── README.md
```

## Installation

### 1. Clone the Repository

``` bash
git clone https://github.com/MaheenAli4566/Drowsiness-Detection-System.git
```

### 2. Open the Project Folder

``` bash
cd Drowsiness-Detection-System
```

### 3. Create a Virtual Environment (Recommended)

``` bash
python -m venv .venv
```

Activate it on Windows:

``` powershell
.venv\\Scripts\\activate
```

### 4. Install Required Libraries

``` bash
python -m pip install numpy opencv-python scipy pygame dlib
```

**Note:** Installing dlib may require CMake and Microsoft C++ Build
Tools on Windows.

### 5. Run the Application

``` bash
python webcam_drowsiness.py
```

## Detection Thresholds

The current application uses the following thresholds:

  Parameter                  Threshold
  -------------------------- ------------
  Eye Aspect Ratio (EAR)     Below 0.21
  Sleepy Eye EAR             Below 0.25
  Mouth Aspect Ratio (MAR)   Above 0.6

These values are used by the application to identify eye closure, sleepy
eyes, and yawning.

## Controls

  Key   Function
  ----- --------------------------
  Q     Quit the application
  R     Reset detection counters

## Detection Results

The application displays detection events in the terminal, including:

-   Total eye blinks
-   Eyes closed events
-   Sleepy-eye count
-   Yawns detected
-   Head nods

A session summary is displayed when the application is closed.

## Requirements

-   Python installed on your computer
-   Webcam
-   Required Python libraries
-   68-point facial landmark predictor file

## Important Note

The facial landmark predictor file
(`shape_predictor_68_face_landmarks.dat`) is required for facial
landmark detection. Keep it in the main project directory.

This project is developed for educational and demonstration purposes. It
should not be considered a replacement for certified driver-safety
systems.

## Author

**Maheen Ali**

GitHub: [@MaheenAli4566](https://github.com/MaheenAli4566)

------------------------------------------------------------------------

⭐ If you find this project useful, consider giving the repository a
star!
