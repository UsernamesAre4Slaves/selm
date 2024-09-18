import React, { useState } from 'react';
import axios from 'axios';
import { Button, Spinner, Alert, Card } from 'react-bootstrap';

function ModelEvaluation() {
    const [result, setResult] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const evaluateModel = async () => {
        setLoading(true);
        setError(null);
        setResult('');

        try {
            // Call the backend to trigger the evaluate.py script directly
            const response = await axios.post('/evaluate_model'); // Modify this to call the backend
            setResult(`Evaluation result: ${response.data.result}`);
        } catch (error) {
            setError('Evaluation failed. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <Card className="p-3 mb-4">
            <Card.Body>
                <Card.Title>Model Evaluation</Card.Title>
                <Button 
                    variant="primary" 
                    onClick={evaluateModel} 
                    disabled={loading}
                >
                    {loading ? <Spinner animation="border" size="sm" /> : 'Evaluate Model'}
                </Button>
                <div className="mt-3">
                    {error && <Alert variant="danger">{error}</Alert>}
                    {result && !loading && <p>{result}</p>}
                </div>
            </Card.Body>
        </Card>
    );
}

export default ModelEvaluation;
