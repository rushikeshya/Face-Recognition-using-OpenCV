# Face Recognition using OpenCv

## Image Example

<img src="Face Recognition demo.png" alt="Example Image" width="600" />



This project implements a face recognition application using OpenCV. The application trains a model using images stored in a dataset directory and then recognizes faces from a live webcam feed.

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Structure](#dataset-structure)
- [Contributing](#contributing)
- [License](#license)

## Description

The application uses the LBPH (Local Binary Patterns Histograms) algorithm for face recognition. It detects faces in images using a pre-trained Haar Cascade classifier, trains the recognizer with labeled images, and then recognizes faces from a live webcam feed.

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/face-recognition-opencv.git
    cd face-recognition-opencv
    ```

2. **Install the required packages:**

    Ensure you have Python 3.x installed. Install the necessary packages using pip:

    ```sh
    pip install numpy opencv-python opencv-contrib-python
    ```

3. **Download the Haar Cascade file:**

    Ensure you have the `haarcascade_frontalface_default.xml` file in your working directory. You can download it from the [OpenCV GitHub repository](https://github.com/opencv/opencv/tree/master/data/haarcascades).

## Usage

1. **Prepare the dataset:**

    Organize your dataset in the following structure:
    ```
    datasets/
    ├── person1/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── person2/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    └── ...
    ```

2. **Run the application:**

    Execute the face recognition script:

    ```sh
    python face_recognition.py
    ```

    The application will train the recognizer with the images in the `datasets` directory and then start recognizing faces from the webcam feed. Press `q` to quit the application.

## Dataset Structure

Ensure your dataset directory is structured correctly with subdirectories for each person, containing their face images:

