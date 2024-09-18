# SELM (Small Efficient Language Model)

**SELM** is a lightweight, modular, and highly efficient language model designed to run on minimal hardware without requiring a GPU. It leverages modern techniques such as knowledge graph integration, task-specific fine-tuning, pruning, and quantization to deliver competitive NLP performance in constrained environments.

![image](https://github.com/user-attachments/assets/7e4c6d44-1801-4a58-9dd3-854223edc817)

## Table of Contents
- [Overview](overview.md)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
  - [Pruning and Quantization](#pruning-and-quantization)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
- [Contributing](#contributing)
- [License](#license)

## Project Structure
```plaintext
├── create_selm_structure.sh
├── SELM
│   ├── backend
│   │   └── server.py
│   ├── config
│   │   ├── active_learning_config.yaml
│   │   ├── distributed_config.yaml
│   │   ├── evaluation_config.yaml
│   │   ├── model_config.yaml
│   │   ├── neuro_evolution_config.yaml
│   │   └── optuna_config.yaml
│   ├── data
│   │   ├── knowledge_graph
│   │   │   └── graph_data.csv
│   │   ├── processed
│   │   ├── raw
│   │   │   └── 'Weather Data.csv'
│   │   └── synthetic
│   ├── frontend
│   │   ├── package.json
│   │   ├── public
│   │   ├── README.md
│   │   └── src
│   │       ├── App.js
│   │       ├── axiosConfig.js
│   │       ├── components
│   │       │   ├── Chatbot.js
│   │       │   ├── ModelEvaluation.js
│   │       │   └── ModelTraining.js
│   │       ├── Dashboard.js
│   │       ├── index.js
│   │       └── styles
│   │           ├── Chatbot.css
│   │           └── index.css
│   ├── notebooks
│   │   ├── dynamic_inference.ipynb
│   │   ├── experiment_1.ipynb
│   │   ├── hyperparameter_search.ipynb
│   │   ├── mixed_precision_experiment.ipynb
│   │   ├── neuro_evolution_experiment.ipynb
│   │   └── pruning_experiment.ipynb
│   ├── scripts
│   │   ├── attention.py
│   │   ├── data_collection.py
│   │   ├── data_ingestion.py
│   │   ├── distributed_inference.py
│   │   ├── dynamic_inference_test.py
│   │   ├── evaluate.py
│   │   ├── neuro_evolution.py
│   │   ├── prune_and_quantize.py
│   │   ├── run_neuro_evolution.py
│   │   └── run_optuna_tuning.py
│   ├── src
│   │   ├── __init__.py
│   │   ├── knowledge_graph
│   │   │   ├── gnn.py
│   │   │   ├── graph_utils.py
│   │   │   ├── __init__.py
│   │   │   └── sparse_gnn.py
│   │   ├── model
│   │   │   ├── attention.py
│   │   │   ├── dynamic_inference.py
│   │   │   ├── embedding.py
│   │   │   ├── __init__.py
│   │   │   ├── output.py
│   │   │   └── transformer.py
│   │   ├── neuro_evolution
│   │   │   ├── crossover.py
│   │   │   ├── fitness.py
│   │   │   ├── mutation.py
│   │   │   ├── neuro_evolution_engine.py
│   │   │   ├── population.py
│   │   │   ├── __init__.py
│   │   │   └── selection.py
│   │   ├── optimization
│   │   │   ├── cache_optimization.py
│   │   │   ├── distributed_training.py
│   │   │   ├── low_rank_factorization.py
│   │   │   ├── mixed_precision.py
│   │   │   ├── __init__.py
│   │   │   └── optuna_tuner.py
│   │   ├── scraper
│   │   │   └── scraper.py
│   │   └── tasks
│   │       ├── active_learning.py
│   │       ├── __init__.py
│   │       ├── question_answering.py
│   │       ├── summarization.py
│   │       └── text_classification.py
│   └── tests
│       ├── test_dynamic_inference.py
│       ├── test_gnn.py
│       ├── test_model.py
│       ├── test_neuro_evolution.py
│       ├── test_optimization.py
│       └── test_tasks.py
└── TEST
    ├── config
    │   └── neuro_evolution_config.yaml
    ├── notebooks
    │   └── neuro_evolution_experiment.ipynb
    ├── scripts
    │   ├── neuro_evolution.py
    │   └── run_neuro_evolution.py
    ├── src
    │   └── neuro_evolution
    │       ├── crossover.py
    │       ├── fitness.py
    │       ├── mutation.py
    │       ├── neuro_evolution_engine.py
    │       ├── population.py
    │       ├── __init__.py
    │       └── selection.py
    └── tests
        └── test_neuro_evolution.py
```


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

## Contributing
#### Contributions are welcome! Please follow these steps:
Fork the repository.
Create a feature branch.
Commit your changes.
Push to the feature branch.
Create a pull request.

### License
This project is licensed under the MIT License - see the LICENSE file for details.



























