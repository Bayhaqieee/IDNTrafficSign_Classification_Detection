# Traffic Sign Classification and Detection Dataset

Welcome to Our Project! This Project is Focusing on Detecting and Classifying Traffic Sign esp. Indonesian Traffic Sign using (currently) **YOLOv11** and **MobileNetV2**

---

## Dataset

Check out our Dataset here:
We Use different dataset to Determine our Model Effectivity
- [Kaggle Dataset 1](https://www.kaggle.com/datasets/ikbal12082004/traffic-sign-in-indonesia)
- [Kaggle Dataset 2](https://www.kaggle.com/datasets/adityabayhaqie/indonesia-traffic-sign-dataset-yolov11/data)

## Project Status

🚧 **Status**: `Ongoing..`

---

## Project Target

- Classification
    - Able to Detecting Traffic Sign
    - The Input will be A Picture of Traffic Sign
    - The Output will be A Text that determine which kind the Traffic Sign is

- Image and Video Detection
    - Able to Detect Traffic Sign through Video and Images by Bounding the Targeted Traffic Signs

---

## Technologies

###Classification
- Python: General-purpose programming language for deep learning and data processing.
- TensorFlow & Keras: Deep learning framework used to build, train, and optimize neural networks.
- MobileNetV2: A lightweight, pre-trained deep learning model optimized for mobile and embedded vision applications.
- Dense, GlobalAveragePooling2D: Layers used in deep learning models for feature extraction and classification.
- Dropout: A regularization technique to prevent overfitting.
- Regularizers: Used for adding penalties to the model to improve generalization.
- EarlyStopping: A callback that stops training when performance stops improving, preventing overfitting.
- TensorFlow Keras ImageDataGenerator: Performs real-time image augmentation to increase dataset diversity.
- OpenCV (cv2): Used for additional image processing, such as resizing, filtering, and transformations.
- KaggleHub & KaggleDatasetAdapter: Enables access to datasets from Kaggle for training and testing.
- Google Colab (files module): Used to upload and download files within Google Colab.
- os & pathlib: Handle file paths and directories for organizing datasets.
- Matplotlib: Used for plotting data such as loss curves, accuracy trends, and dataset samples.

###Detection
- **Python**: General-purpose programming language used for data processing, visualization, and machine learning.
- **NumPy**: Supports numerical computations, including array manipulations and mathematical operations.
- **Pandas**: Handles structured data, such as reading, cleaning, and analyzing tabular datasets.
- **OpenCV (cv2)**: Provides computer vision capabilities, including image processing, object detection, and video analysis.
- **PIL (Pillow)**: Image processing library for handling various image formats and transformations.
- **Ultralytics YOLO**: Deep learning-based object detection framework, used for real-time object recognition.
- **Matplotlib & Seaborn**: Visualization libraries for creating graphs and statistical plots.
- **IPython.display (Video)**: Used for displaying videos within Jupyter notebooks.
- **Pathlib & Glob**: File and directory management tools for handling file paths and searching for specific file patterns.
- **tqdm**: Provides progress bars to track loop execution, useful in data processing and model training.
- **Warnings**: Used to manage and suppress unnecessary warnings in Python scripts.
- **Optuna**: Hyperparameter optimization framework for improving machine learning model performance.

---

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/Bayhaqieee/traffic_sign_classification-detection.git
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Notebeooks, Each of them already Labeled with `Classification` or `Detection`
