import React, { useState, useEffect, useRef } from 'react';
import './Chatbot.css'; // Add a CSS file to style the Chatbot component
import axios from 'axios';

const Chatbot = () => {
  const [messages, setMessages] = useState(() => {
    // Load chat history from local storage if available
    const savedMessages = localStorage.getItem('chatMessages');
    return savedMessages ? JSON.parse(savedMessages) : [];
  });
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatEndRef = useRef(null);

  // Save messages to local storage whenever they change
  useEffect(() => {
    localStorage.setItem('chatMessages', JSON.stringify(messages));
  }, [messages]);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleInputChange = (e) => {
    setInput(e.target.value);
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { sender: 'user', text: input };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setInput('');

    setIsLoading(true);
    try {
      const response = await axios.post('/api/chatbot', { message: input });
      const botMessage = { sender: 'bot', text: response.data.reply };
      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error) {
      const errorMessage = {
        sender: 'bot',
        text: 'Sorry, something went wrong. Try again later.',
      };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    }
    setIsLoading(false);
  };

  // Quick reply feature
  const handleQuickReply = (reply) => {
    setInput(reply);
    handleSendMessage({ preventDefault: () => {} }); // Automatically submit the quick reply
  };

  return (
    <div className="chatbot-container">
      <div className="chatbot-header">
        <h2>AI Chatbot</h2>
      </div>

      <div className="chatbot-body">
        <div className="chat-history">
          {messages.map((msg, index) => (
            <div key={index} className={`chat-message ${msg.sender}`}>
              <div className="message-bubble">{msg.text}</div>
            </div>
          ))}
          {isLoading && (
            <div className="chat-message bot">
              <div className="message-bubble loading">Thinking...</div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>
      </div>

      <div className="quick-replies">
        {/* Example Quick Replies */}
        <button onClick={() => handleQuickReply('Hello!')}>Hello!</button>
        <button onClick={() => handleQuickReply('Tell me a joke')}>Tell me a joke</button>
      </div>

      <div className="chatbot-footer">
        <form onSubmit={handleSendMessage}>
          <input
            type="text"
            className="chat-input"
            placeholder="Type your message..."
            value={input}
            onChange={handleInputChange}
            disabled={isLoading}
          />
          <button type="submit" className="send-button" disabled={isLoading}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
};

export default Chatbot;
