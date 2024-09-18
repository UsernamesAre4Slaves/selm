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
