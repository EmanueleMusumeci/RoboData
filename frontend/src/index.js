import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

// Optional: Only import reportWebVitals if the file exists
try {
  // This will be replaced by webpack with a boolean during build
  if (process.env.NODE_ENV === 'production') {
    const reportWebVitals = require('./reportWebVitals').default;
    reportWebVitals();
  }
} catch (e) {
  console.log('Web Vitals reporting is not available');
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
