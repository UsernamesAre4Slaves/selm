import axios from 'axios';

const instance = axios.create({
    baseURL: 'http://localhost:5000/api', // Ensure this URL matches your backend server
});

export default instance;
