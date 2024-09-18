import React, { useState, useEffect } from 'react';
import axios from '../axiosConfig';
import { Card, Spinner, Alert, ListGroup, Button, Modal } from 'react-bootstrap';
import Chatbot from './components/Chatbot';

const Dashboard = () => {
    const [metrics, setMetrics] = useState({});
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [showTrainingModal, setShowTrainingModal] = useState(false);
    const [trainingStatus, setTrainingStatus] = useState('');
    const [trainingLoading, setTrainingLoading] = useState(false);
    const [chatHistory, setChatHistory] = useState([]);

    useEffect(() => {
        const fetchMetrics = async () => {
            try {
                const response = await axios.get('/metrics');
                setMetrics(response.data);
            } catch (error) {
                setError('Failed to fetch metrics.');
            } finally {
                setLoading(false);
            }
        };

        fetchMetrics();
    }, []);

    const startChatbotTraining = async () => {
        setTrainingLoading(true);
        setTrainingStatus('Training started...');
        try {
            const response = await axios.post('/api/train', { chatbot: true });
            setTrainingStatus(`Training completed! Model ID: ${response.data.modelId}`);
        } catch (error) {
            setTrainingStatus('Training failed. Please try again.');
        } finally {
            setTrainingLoading(false);
        }
    };

    const handleClose = () => setShowTrainingModal(false);
    const handleShow = () => setShowTrainingModal(true);

    if (loading) {
        return (
            <div className="d-flex justify-content-center align-items-center" style={{ height: '100vh' }}>
                <Spinner animation="border" />
                <span className="ms-3">Loading metrics...</span>
            </div>
        );
    }

    return (
        <div>
            <h1>Dashboard</h1>
            {error && <Alert variant="danger">{error}</Alert>}

            {/* Metrics Overview */}
            {Object.keys(metrics).length > 0 ? (
                <Card className="p-3 mb-4">
                    <Card.Body>
                        <Card.Title>Model Metrics Overview</Card.Title>
                        <ListGroup>
                            {Object.entries(metrics).map(([key, value]) => (
                                <ListGroup.Item key={key}>
                                    <strong>{key}:</strong> {value}
                                </ListGroup.Item>
                            ))}
                        </ListGroup>
                    </Card.Body>
                </Card>
            ) : (
                <p>No metrics available.</p>
            )}

            {/* Chatbot Interaction */}
            <Card className="p-3 mb-4">
                <Card.Body>
                    <Card.Title>Chatbot Interaction</Card.Title>
                    <Chatbot chatHistory={chatHistory} setChatHistory={setChatHistory} />
                </Card.Body>
            </Card>

            {/* Chatbot Training */}
            <Button variant="primary" onClick={handleShow}>
                Train Chatbot Model
            </Button>

            <Modal show={showTrainingModal} onHide={handleClose}>
                <Modal.Header closeButton>
                    <Modal.Title>Train Chatbot Model</Modal.Title>
                </Modal.Header>
                <Modal.Body>
                    {trainingLoading ? (
                        <div className="d-flex justify-content-center align-items-center">
                            <Spinner animation="border" />
                            <span className="ms-3">Training in progress...</span>
                        </div>
                    ) : (
                        <>
                            <p>Click the button below to start training the chatbot model.</p>
                            <Button variant="primary" onClick={startChatbotTraining} disabled={trainingLoading}>
                                {trainingLoading ? <Spinner animation="border" size="sm" /> : 'Start Training'}
                            </Button>
                            {trainingStatus && <Alert className="mt-3" variant={trainingStatus.includes('failed') ? 'danger' : 'success'}>{trainingStatus}</Alert>}
                        </>
                    )}
                </Modal.Body>
                <Modal.Footer>
                    <Button variant="secondary" onClick={handleClose}>
                        Close
                    </Button>
                </Modal.Footer>
            </Modal>
        </div>
    );
};

export default Dashboard;
