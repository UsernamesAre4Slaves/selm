# SELM (Small Efficient Language Model)

**SELM** is a lightweight, modular, and highly efficient language model designed to run on minimal hardware without requiring a GPU. It leverages modern techniques such as knowledge graph integration, task-specific fine-tuning, pruning, and quantization to deliver competitive NLP performance in constrained environments.

![image](https://github.com/user-attachments/assets/7e4c6d44-1801-4a58-9dd3-854223edc817)


## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
  - [Pruning and Quantization](#pruning-and-quantization)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
- [Tasks](#tasks)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Lightweight Transformer Model**: Compact architecture designed to run efficiently on CPUs.
- **Modular Design**: The model is built in a modular fashion, allowing easy swapping of components (e.g., tokenizers, attention mechanisms).
- **Task-Specific Fine-Tuning**: Support for text classification, summarization, and question-answering tasks.
- **Knowledge Graph Integration**: Utilize a Graph Neural Network (GNN) to integrate knowledge graphs for improved fact-based reasoning.
- **Model Optimization**: Supports pruning and quantization to reduce model size and improve inference efficiency.
- **Automated Hyperparameter Tuning**: Use Optuna for automated tuning of the model architecture and training parameters.

## Project Structure
<div style="overflow-y: scroll; height: 200px; width: 100%; padding: 10px; border: 1px solid #ccc;">
  <pre><code>
SELM/
│
├── src/                        # Source code for SELM
│   ├── __init__.py             # Init file to make `src` a package
│   ├── model/                  # Core language model modules
│   │   ├── __init__.py
│   │   ├── transformer.py      # Transformer architecture (main model)
│   │   ├── embedding.py        # Embedding layer module
│   │   ├── tokenization.py     # Tokenization module (e.g., SentencePiece, BPE)
│   │   ├── attention.py        # Attention mechanism (e.g., Linformer, Performer)
│   │   ├── output.py           # Task-specific output heads (e.g., classification, generation)
│   │   └── dynamic_inference.py # Module for dynamic inference (early exits, conditional compute)
│   │
│   ├── optimization/           # Hyperparameter tuning and model optimization
│   │   ├── __init__.py
│   │   ├── optuna_tuner.py     # Script for hyperparameter tuning with Optuna
│   │   ├── pruning.py          # Pruning and quantization modules
│   │   ├── mixed_precision.py  # Mixed-precision training module
│   │   ├── low_rank_factorization.py # Module for low-rank matrix factorization
│   │   ├── distributed_training.py # Distributed model training for large datasets
│   │   └── cache_optimization.py  # Cache-aware optimization for inference and training
│   │
│   ├── knowledge_graph/        # Knowledge graph integration for specific tasks
│   │   ├── __init__.py
│   │   ├── graph_utils.py      # Utilities for handling graphs (e.g., loading, querying)
│   │   ├── gnn.py              # GNN architecture for knowledge graph-based tasks
│   │   └── sparse_gnn.py       # Sparse GNN implementation for memory efficiency
│   │
│   └── tasks/                  # Task-specific modules for fine-tuning
│       ├── __init__.py
│       ├── text_classification.py  # Fine-tuning for text classification tasks
│       ├── summarization.py        # Fine-tuning for text summarization
│       ├── question_answering.py   # Fine-tuning for question-answering tasks
│       └── active_learning.py      # Active learning module for data-efficient training
│
├── scripts/                    # Scripts for running experiments, training, etc.
│   ├── train.py                # Main training script for the model
│   ├── evaluate.py             # Evaluation script to benchmark the model
│   ├── prune_and_quantize.py   # Script to apply pruning and quantization
│   ├── run_optuna_tuning.py    # Script for running Optuna hyperparameter search
│   ├── distributed_inference.py # Script for inference across distributed environments
│   └── dynamic_inference_test.py # Script for testing dynamic inference mechanisms
│
├── config/                     # Configuration files (e.g., YAML, JSON)
│   ├── model_config.yaml       # Model architecture configuration (e.g., layers, heads)
│   ├── training_config.yaml    # Training-related configurations (batch size, epochs, etc.)
│   ├── optuna_config.yaml      # Configuration for hyperparameter tuning
│   ├── active_learning_config.yaml # Configurations for active learning sampling
│   └── distributed_config.yaml # Configuration for distributed training and inference
│
├── data/                       # Directory for datasets (can be symlinked to save space)
│   ├── raw/                    # Raw datasets
│   ├── processed/              # Preprocessed data files
│   ├── knowledge_graph/        # Knowledge graph data files (e.g., RDF, CSV)
│   └── synthetic/              # Generated synthetic data for augmentation
│
├── tests/                      # Unit tests and integration tests
│   ├── test_model.py           # Tests for the model components
│   ├── test_tasks.py           # Tests for task-specific modules
│   ├── test_optimization.py    # Tests for optimization (pruning, Optuna, mixed precision)
│   ├── test_gnn.py             # Tests for knowledge graph and GNN integration
│   └── test_dynamic_inference.py # Tests for dynamic inference and conditional computation
│
├── notebooks/                  # Jupyter notebooks for experiments and prototyping
│   ├── experiment_1.ipynb      # Example notebook for model testing or development
│   ├── hyperparameter_search.ipynb # Notebook for Optuna-based tuning exploration
│   ├── pruning_experiment.ipynb    # Example of pruning/quantization experiments
│   ├── dynamic_inference.ipynb     # Experimenting with dynamic inference strategies
│   └── mixed_precision_experiment.ipynb # Notebook for mixed-precision training results
│
├── requirements.txt            # Python dependencies list
├── README.md                   # Project overview and setup instructions
├── LICENSE                     # License file for open-source use (MIT, Apache, etc.)
├── setup.py                    # Python package setup script for the SELM project
└── .gitignore                  # Ignore specific files from version control
  </code></pre>
</div>


## Installation
To set up the SELM project, follow these steps:

### Prerequisites
- Python 3.7 or later
- `pip` (Python package installer)
- Node.js and npm (for front-end)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/UsernamesAre4Slaves/SELM.git
   cd SELM
   
2. Install Python dependencies:
   <code>pip install -r requirements.txt</code>

3. Install Node.js dependencies for the front-end:
   <code>cd frontend</code>
   <code>npm install</code>

### Usage
## Training the Model
To train the model, start the back-end server and send a request to the training endpoint:

1. Start the back-end server:
   <code>cd backend</code>
   <code>python server.py</code>

2. Trigger training via the front-end or directly using:
   <code>curl -X POST http://localhost:5000/api/train</code>

## Evaluating the Model
To evaluate the model, send a request to the evaluation endpoint:
<code>curl -X GET http://localhost:5000/api/evaluate</code>

## Pruning and Quantization
<code>curl -X POST http://localhost:5000/api/prune_and_quantize</code>

## Hyperparameter Tuning
To run Optuna hyperparameter tuning:
<code>curl -X POST http://localhost:5000/api/optimize</code>

### SELM supports the following tasks:
Text Classification: Classify text into predefined categories.
Summarization: Generate summaries of longer texts.
Question Answering: Answer questions based on provided context.

### Contributions are welcome! Please follow these steps:
Fork the repository.
Create a feature branch.
Commit your changes.
Push to the feature branch.
Create a pull request.

### License
This project is licensed under the MIT License - see the LICENSE file for details.



























