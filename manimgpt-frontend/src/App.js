// Importing necessary libraries
import React, { useState } from 'react';
import './App.css'; // For custom styling, including the dark theme
import { motion } from 'framer-motion';
import ClipLoader from 'react-spinners/ClipLoader';

// Main App Component
const App = () => {
  // State for storing the prompt input and video URL
  const [prompt, setPrompt] = useState('');
  const [videoUrl, setVideoUrl] = useState(null);
  const [loading, setLoading] = useState(false);

  // Handler for prompt input change
  const handleInputChange = (e) => {
    setPrompt(e.target.value);
  };

  // Submit button handler
  const handleSubmit = async () => {
    if (!prompt) return;
    setLoading(true);

    // Send the prompt to the backend
    try {
      const response = await fetch('https://your-backend-url.com/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });

      const data = await response.json();
      setVideoUrl(data.videoUrl);
    } catch (error) {
      console.error('Error generating video:', error);
      setVideoUrl(null);
    }

    setLoading(false);
  };

  return (
    <div className="app-container">
      <motion.header className="app-header" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 1 }}>
        <h1>ManimGPT</h1>
        <p>Turn complex research into beautiful animations effortlessly</p>
      </motion.header>

      <motion.main className="main-content" initial={{ y: -20 }} animate={{ y: 0 }} transition={{ duration: 0.5 }}>
        <div className="prompt-section">
          <motion.input
            type="text"
            placeholder="Enter your prompt here..."
            value={prompt}
            onChange={handleInputChange}
            className="prompt-input"
            whileFocus={{ scale: 1.05 }}
          />
          <motion.button 
            className="submit-button" 
            onClick={handleSubmit} 
            disabled={loading}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {loading ? 'Generating...' : 'Generate Animation'}
          </motion.button>
        </div>

        {loading && (
          <div className="loading-indicator">
            <ClipLoader color="#ff4081" loading={loading} size={50} />
            <p className="loading-text">Please wait while we generate your animation...</p>
          </div>
        )}

        {videoUrl && (
          <motion.div className="video-container" initial={{ scale: 0.8 }} animate={{ scale: 1 }} transition={{ duration: 0.5 }}>
            <video controls>
              <source src={videoUrl} type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          </motion.div>
        )}
      </motion.main>
    </div>
  );
};

export default App;
