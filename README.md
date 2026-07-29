# Wildlife Animal Multiclass Classification CNN

Training and evaluation pipeline for classifying wildlife images into multiple species using a convolutional neural network (CNN). This repository provides a complete workflow including data preprocessing, model architecture implementation (MobileNet3-small), training loops with comprehensive loss/accuracy tracking, and visualization of results.

## Project Overview

This project implements a high-performance wildlife classification system using a Convolutional Neural Network (CNN). By leveraging the MobileNet3-small architecture, the pipeline is optimized for both accuracy and computational efficiency in identifying various animal species from image data.

## Getting Started

### Installation

To set up the environment on Windows, follow these steps:

1. **Download the Setup Script**: Obtain the `windows_setup.bat` file from the repository.
2. **Deployment**: Place the script in your desired project directory.
3. **Execution**: Run the script by double-clicking it. Ensure you have administrative privileges on your machine, as the installer will automatically configure all necessary dependencies and environments.

### Troubleshooting

If the automated setup fails, please ensure the following components are manually installed:

*   **Python 3.x**: Ensure Python is added to your system PATH.
*   **PyTorch & Torchvision**: Install via `pip install torch torchvision`.
*   **Dependencies**: Manually install required libraries using:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

Once the environment is configured, you can run the following commands:

*   **Application**: Run the web interface using:
    ```bash
    python Wildlife_Animal_Classifier/app.py
    ```
*   **Training**: Execute the training pipeline via Jupyter Notebook:
    ```bash
    python Wildlife_Animal_Classifier/MulticlassCNN_training_final.ipynb
    ```