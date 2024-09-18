import React, { useState } from 'react';
import axios from 'axios';
import { Button, Spinner, Alert, Card } from 'react-bootstrap';

function ModelTraining() {
    const [status, setStatus] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const startTraining = async () => {
        setLoading(true);
        setStatus('Chatbot-specific training started...');
        setError(null);
        try {
            // Trigger backend route that starts chatbot-specific training
            const response = await axios.post('/start_chatbot_training');
            setStatus(`Training completed successfully! Model ID: ${response.data.modelId}`);
        } catch (error) {
            setError('Training failed. Please try again.');
            setStatus('');
        } finally {
            setLoading(false);
        }
    };

    return (
        <Card className="p-3 mb-4">
            <Card.Body>
                <Card.Title>Model Training</Card.Title>
                <Button 
                    variant="primary" 
                    onClick={startTraining} 
                    disabled={loading}
                >
                    {loading ? <Spinner animation="border" size="sm" /> : 'Start Chatbot Training'}
                </Button>
                <div className="mt-3">
                    {error && <Alert variant="danger">{error}</Alert>}
                    {status && !loading && <p>{status}</p>}
                </div>
            </Card.Body>
        </Card>
    );
}

export default ModelTraining;
