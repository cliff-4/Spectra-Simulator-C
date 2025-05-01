# Detailed Documentation for train.py

## Overview

This script implements a deep learning training pipeline using PyTorch. It is designed to train a CNN-LSTM-Dense model for multi-parameter regression and classification tasks. The script supports distributed training using MPI (Message Passing Interface) and includes functionality for checkpointing, restoring models, and logging training progress.

---

## Table of Contents

1. Dependencies
2. Key Features
3. Script Workflow
4. Functions
   - mkdir_p
   - scale_parameter
   - categorize
5. Classes
   - Model
6. Training Pipeline
   - Distributed Training Setup
   - Data Preparation
   - Training Loop
   - Validation Loop
   - Checkpointing
7. Command-Line Arguments
8. Configuration Parameters
9. Output
10. How to Run

---

## Dependencies

The script requires the following Python libraries:

- `numpy`
- `matplotlib`
- `torch`
- `torch.nn`
- `torch.optim`
- `argparse`
- `os`
- `errno`

Optional dependencies for distributed training:
- `torch_mpi`
- `torch_ccl`

---

## Key Features

1. **Distributed Training**: Supports multi-GPU and multi-node training using MPI.
2. **Custom Model**: Implements a CNN-LSTM-Dense architecture for multi-parameter prediction.
3. **Dynamic Binning**: Converts continuous labels into discrete bins for classification tasks.
4. **Checkpointing**: Saves model checkpoints during training and supports restoring from checkpoints.
5. **Scalable Data Loading**: Handles large datasets using NumPy arrays and distributed data loading.
6. **Learning Rate Scheduler**: Uses `OneCycleLR` for dynamic learning rate adjustment.
7. **Logging**: Logs training and validation loss, accuracy, and time per iteration.

---

## Script Workflow

1. Parse command-line arguments for model path and checkpoint restoration.
2. Initialize distributed training if applicable.
3. Define the CNN-LSTM-Dense model.
4. Load and preprocess training and validation datasets.
5. Train the model over multiple epochs.
6. Validate the model after each epoch.
7. Save checkpoints and the final model.

---

## Functions

### `mkdir_p(path)`
Creates a directory if it does not exist.

**Parameters**:
- `path` (str): Path to the directory.

**Usage**:
```python
mkdir_p('/path/to/directory')
```

---

### `scale_parameter(a)`
Scales a parameter to the range [0, 1].

**Parameters**:
- `a` (numpy array): Input array to scale.

**Returns**:
- Scaled array.

**Usage**:
```python
scaled_array = scale_parameter(array)
```

---

### `categorize(par, num_cl)`
Categorizes a scaled parameter into discrete classes.

**Parameters**:
- `par` (numpy array): Scaled parameter.
- `num_cl` (int): Number of classes.

**Returns**:
- Categorized array.

**Usage**:
```python
categorized_array = categorize(scaled_array, num_classes)
```

---

## Classes

### `Model`

This class defines a CNN-LSTM-Dense architecture for multi-parameter prediction.

#### **Constructor**
```python
def __init__(self, num_classes_dnu, num_classes_dp, num_classes_q, num_classes_acr, num_classes_aer, num_classes_a3, num_classes_inc, num_classes_epp, num_classes_epg, num_classes_numax, num_classes_snr, num_classes_gamma, num_classes_vl1, num_classes_vl2, num_classes_vl3, shape)
```

**Parameters**:
- `num_classes_*`: Number of classes for each output parameter.
- `shape` (int): Input data shape.

#### **Methods**

1. **`forward(x)`**:
   Defines the forward pass of the model.

2. **`size_after_cnnlstm(x)`**:
   Computes the size of the output after the CNN-LSTM layers.

3. **`linear_input_neurons(shape)`**:
   Calculates the number of input neurons for the linear layer.

---

## Training Pipeline

### Distributed Training Setup

- Checks if MPI is available using the `PMI_SIZE` environment variable.
- Initializes the process group for distributed training.
- Sets the device to `cuda` if available, otherwise `cpu`.

---

### Data Preparation

- Loads data from `.npy` files.
- Filters data based on specific conditions.
- Converts continuous labels into discrete bins using `np.digitize`.
- Prepares data loaders for training and validation.

---

### Training Loop

- Iterates over the dataset for a specified number of epochs.
- Computes the loss for each output parameter.
- Logs training loss and time per iteration.
- Updates model weights using the Adam optimizer.

---

### Validation Loop

- Evaluates the model on the validation dataset.
- Computes validation loss and accuracy for each output parameter.
- Logs validation metrics.

---

### Checkpointing

- Saves model checkpoints after each epoch.
- Restores model state from a checkpoint if specified.

---

## Command-Line Arguments

1. `--path`: Path to save model checkpoints.
2. `--restore`: Whether to restore from a checkpoint (`yes` or `no`).

**Example**:
```bash
python train.py --path /path/to/checkpoints --restore yes
```

---

## Configuration Parameters

- `num_epochs`: Number of training epochs.
- `num_batchsize`: Batch size for training.
- `loss`: Loss function (default: `sparse_categorical_crossentropy`).
- `pos_enc`: Whether to use positional encoding (default: `False`).
- `data_dir`: Directory containing the training data.

---

## Output

1. **Logs**:
   - Training and validation loss.
   - Accuracy for each output parameter.
   - Time per iteration.

2. **Checkpoints**:
   - Saved in the specified path.
   - Includes model state, optimizer state, and scheduler state.

3. **Final Model**:
   - Saved as `model.pth` in the specified path.

---

## How to Run

1. Ensure all dependencies are installed.
2. Prepare the training data in `.npy` format.
3. Run the script with the desired arguments:
   ```bash
   python train.py --path /path/to/checkpoints --restore no
   ```
4. Monitor the logs for training progress.

---

This documentation provides a comprehensive overview of the script, its functionality, and how to use it effectively.