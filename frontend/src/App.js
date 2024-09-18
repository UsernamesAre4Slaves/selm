import React, { useState } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import { Container, Row, Col, Button, Alert, Spinner } from 'react-bootstrap';
import ModelTraining from './components/ModelTraining';
import ModelEvaluation from './components/ModelEvaluation';
import Chatbot from './components/Chatbot'; // Import the Chatbot component
import './App.css'; // Custom CSS file for additional styling

function App() {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleTraining = async () => {
        setLoading(true);
        try {
            // Call the API to start training
            await fetch('/api/train', { method: 'POST' });
            setError(null);
        } catch (e) {
            setError('Failed to start training. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const handleEvaluation = async () => {
        setLoading(true);
        try {
            // Call the API to start evaluation
            await fetch('/api/evaluate', { method: 'POST' });
            setError(null);
        } catch (e) {
            setError('Failed to start evaluation. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <Container className="mt-4">
            <Row className="mb-4">
                <Col>
                    <h1 className="text-center">SELM Interface</h1>
                </Col>
            </Row>
            <Row className="mb-4">
                <Col>
                    <Button 
                        variant="primary" 
                        onClick={handleTraining} 
                        disabled={loading}
                    >
                        {loading ? <Spinner animation="border" size="sm" /> : 'Start Training'}
                    </Button>
                    <Button 
                        variant="secondary" 
                        onClick={handleEvaluation} 
                        disabled={loading} 
                        className="ml-2"
                    >
                        {loading ? <Spinner animation="border" size="sm" /> : 'Start Evaluation'}
                    </Button>
                </Col>
            </Row>
            {error && (
                <Row className="mb-4">
                    <Col>
                        <Alert variant="danger">{error}</Alert>
                    </Col>
                </Row>
            )}
            <Row className="mb-4">
                <Col md={6}>
                    <ModelTraining />
                </Col>
                <Col md={6}>
                    <ModelEvaluation />
                </Col>
            </Row>
            <Row>
                <Col>
                    <Chatbot /> {/* Add the Chatbot component here */}
                </Col>
            </Row>
        </Container>
    );
}

export default App;
