from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/api/train', methods=['POST'])
def train_model():
    """Endpoint to start model training."""
    try:
        result = subprocess.run(['python', 'scripts/train.py'], capture_output=True, text=True, check=True)
        return jsonify({'message': 'Training started', 'output': result.stdout})
    except subprocess.CalledProcessError as e:
        return jsonify({'message': 'Training failed', 'error': e.stderr}), 500

@app.route('/api/evaluate', methods=['GET'])
def evaluate_model():
    """Endpoint to evaluate the model."""
    try:
        result = subprocess.run(['python', 'scripts/evaluate.py'], capture_output=True, text=True, check=True)
        return jsonify({'message': 'Evaluation completed', 'result': result.stdout})
    except subprocess.CalledProcessError as e:
        return jsonify({'message': 'Evaluation failed', 'error': e.stderr}), 500

@app.route('/api/prune_and_quantize', methods=['POST'])
def prune_and_quantize():
    """Endpoint to apply pruning and quantization."""
    try:
        result = subprocess.run(['python', 'scripts/prune_and_quantize.py'], capture_output=True, text=True, check=True)
        return jsonify({'message': 'Pruning and quantization completed', 'output': result.stdout})
    except subprocess.CalledProcessError as e:
        return jsonify({'message': 'Pruning and quantization failed', 'error': e.stderr}), 500

@app.route('/api/optimize', methods=['POST'])
def run_optuna_tuning():
    """Endpoint to start Optuna hyperparameter tuning."""
    try:
        result = subprocess.run(['python', 'scripts/run_optuna_tuning.py'], capture_output=True, text=True, check=True)
        return jsonify({'message': 'Optuna tuning completed', 'output': result.stdout})
    except subprocess.CalledProcessError as e:
        return jsonify({'message': 'Optuna tuning failed', 'error': e.stderr}), 500

if __name__ == '__main__':
    app.run(port=5000)
