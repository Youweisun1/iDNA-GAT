### Overview

MeTT predicts 6mA sites from 41-bp DNA sequences. It combines sequence and graph views and uses gradient affinity to select auxiliary tasks for cross-task learning.

The workflow is as follows:

1. Each species-specific dataset is treated as a binary classification task and split into 90% training data and 10% test data using a fixed seed.
2. The sequence view is represented by a one-hot matrix, whereas graph nodes use three-dimensional nucleotide chemical properties.
3. The graph contains bidirectional edges at index distances of 1, 2, 4, and 8, together with bidirectional edges between the candidate center and all sequence positions.
4. A CNN extracts local sequence features, a three-layer residual GAT extracts multiscale graph features, and residual multi-head self-attention integrates the fused representation.
5. Stage I selects a low-conflict auxiliary task from training-gradient affinity and jointly trains it with the target task.
6. Stage II starts from the best Stage I parameters and refines the model using only the target task.

### Project structure

```text
MeTT/
├── ModelTraining2.py   # Training and evaluation entry point
├── MyModel.py          # CNN, GAT, feature fusion, and classifier
├── Bert_network2.py    # Residual multi-head self-attention
├── data_processing5.py # XLSX loading, splitting, encoding, and graph building
├── requirements.txt    # Python dependencies
├── checkpoints/        # Saved model parameters
└── results/            # Metrics and task-selection records

../data7/               # Eleven 6mA datasets in XLSX format
```

### Environment

The project has been tested with:

```text
Python 3.8.19
PyTorch 2.0.0 + CUDA 11.8
PyTorch Geometric 2.6.1
```

Install the remaining dependencies with:

```bash
pip install -r requirements.txt
```

PyTorch and PyTorch Geometric must match the local CUDA version.

### Training

Edit the main settings at the top of `ModelTraining.py`:

```python
Target = "A. thaliana"  # Target dataset
Checkpoint = ""         # Leave empty for training
Graphmode = "multiscale"
Seed = 1377
Dynamicepochs = 50       # Stage I
Refinementepochs = 20    # Stage II
Batch = 64
Multidomainlr = 0.001
Refinementlr = 0.0001
Auxiliaryweight = 0.5
```
