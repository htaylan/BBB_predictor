# BBB Classifier

This project contains a deep neural network classifier implemented using PyTorch. The classifier is used for binary classification tasks (can pass, or can not pass BBB)

## Project Structure

- `data`: Directory to store dataset files.
- `models`: Directory containing model architecture and classifier implementation.
- `utils`: Directory containing data preprocessing and metrics utility functions.
- `main.py`: Main script to run the training and evaluation.
- `requirements.txt`: List of Python dependencies.
- `README.md`: Project documentation.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/htaylan/BBB_predictor.git
    cd BBB_predictor
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Place your dataset files (`descriptors.npy` and `labels.npy`) in the `data` directory.

## Usage

To train the model, run:

```bash
python train.py
```

To evaluate your molecules, run:

```
python predict.py
```

