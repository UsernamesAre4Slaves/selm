# SELM (Small Efficient Language Model)

**SELM** is a lightweight, modular, and highly efficient language model designed to run on minimal hardware without requiring a GPU. It leverages modern techniques such as knowledge graph integration, task-specific fine-tuning, pruning, and quantization to deliver competitive NLP performance in constrained environments. SELM is inspired by minimalist Linux distributions, focusing on efficiency and compact design.

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
The SELM project is organized into the following directories and files:
SELM/ ├── backend/ # Back-end server code │ ├── server.py # Flask server to handle API requests │ └── requirements.txt # Python dependencies for the back-end │ ├── frontend/ # Front-end web application │ ├── public/ # Static files (e.g., index.html) │ ├── src/ # Source code for front-end │ │ ├── components/ # React components │ │ ├── App.js # Main application component │ │ ├── index.js # Entry point for React │ │ ├── styles/ # CSS or styling files │ │ └── axiosConfig.js # Axios configuration for API calls │ ├── package.json # Dependencies and scripts │ └── README.md # Front-end specific documentation │ ├── src/ # Source code for SELM │ ├── init.py │ ├── model/ │ ├── optimization/ │ ├── knowledge_graph/ │ └── tasks/ │ ├── scripts/ # Scripts for running experiments, training, etc. │ ├── config/ # Configuration files │ ├── data/ # Directory for datasets │ ├── tests/ # Unit tests and integration tests │ ├── notebooks/ # Jupyter notebooks for experiments │ ├── requirements.txt # Python dependencies list ├── README.md # Project overview and setup instructions ├── setup.py # Python package setup script ├── LICENSE # License file └── .gitignore # Ignore specific files from version control


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
   pip install -r requirements.txt

3. Install Node.js dependencies for the front-end:
   cd frontend
   npm install

### Usage
## Training the Model
To train the model, start the back-end server and send a request to the training endpoint:

1. Start the back-end server:
   cd backend
   python server.py

2. Trigger training via the front-end or directly using:
   curl -X POST http://localhost:5000/api/train

## Evaluating the Model
To evaluate the model, send a request to the evaluation endpoint:
curl -X GET http://localhost:5000/api/evaluate

## Pruning and Quantization
curl -X POST http://localhost:5000/api/prune_and_quantize

## Hyperparameter Tuning
To run Optuna hyperparameter tuning:
curl -X POST http://localhost:5000/api/optimize

### Tasks

## SELM supports the following tasks:

    Text Classification: Classify text into predefined categories.
    Summarization: Generate summaries of longer texts.
    Question Answering: Answer questions based on provided context.

### Contributing

## Contributions are welcome! Please follow these steps:

    Fork the repository.
    Create a feature branch.
    Commit your changes.
    Push to the feature branch.
    Create a pull request.

### License
This project is licensed under the MIT License - see the LICENSE file for details.



























