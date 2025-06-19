#!/bin/bash

# Change to project root directory
cd "$(dirname "$0")"

# Check if frontend directory exists, if not create it
if [ ! -d "frontend" ]; then
    echo "Creating frontend directory..."
    npx create-react-app frontend
    cd frontend
else
    # Navigate to frontend directory
    cd frontend
    
    # If package.json doesn't exist, initialize a new React app
    if [ ! -f "package.json" ]; then
        echo "Initializing new React app..."
        npx create-react-app .
    fi
fi

# Install dependencies
echo "Installing/updating dependencies..."
npm install

# Create or update public/index.html if it doesn't exist
mkdir -p public
if [ ! -f "public/index.html" ]; then
    echo "Creating default index.html..."
    cat > public/index.html << 'EOL'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="RoboData Frontend" />
    <title>RoboData</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
EOL
fi

# Create or update src/App.js if it doesn't exist
mkdir -p src
if [ ! -f "src/App.js" ]; then
    echo "Creating default App.js..."
    cat > src/App.js << 'EOL'
import React from 'react';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Welcome to RoboData</h1>
        <p>Frontend is running successfully!</p>
      </header>
    </div>
  );
}

export default App;
EOL
fi

# Create or update src/index.js if it doesn't exist
if [ ! -f "src/index.js" ]; then
    echo "Creating default index.js..."
    cat > src/index.js << 'EOL'
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
EOL
fi

# Create or update src/App.css if it doesn't exist
if [ ! -f "src/App.css" ]; then
    echo "Creating default App.css..."
    cat > src/App.css << 'EOL'
.App {
  text-align: center;
}

.App-logo {
  height: 40vmin;
  pointer-events: none;
}

@media (prefers-reduced-motion: no-preference) {
  .App-logo {
    animation: App-logo-spin infinite 20s linear;
  }
}

.App-header {
  background-color: #282c34;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  font-size: calc(10px + 2vmin);
  color: white;
}

.App-link {
  color: #61dafb;
}

@keyframes App-logo-spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}
EOL
fi

# Create or update src/index.css if it doesn't exist
if [ ! -f "src/index.css" ]; then
    echo "Creating default index.css..."
    cat > src/index.css << 'EOL'
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
    monospace;
}
EOL
fi

# Start the development server
echo "Starting frontend development server..."
npm start
