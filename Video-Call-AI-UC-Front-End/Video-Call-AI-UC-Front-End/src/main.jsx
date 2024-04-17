import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import StatusProvider from './component/StatusProvider';
import * as sdk from 'microsoft-cognitiveservices-speech-sdk';

// sdk.Diagnostics.SetLoggingLevel(sdk.LogLevel.Debug);

const container = document.getElementById('app');
const root = createRoot(container);

root.render(
  <StatusProvider>
    <App />
  </StatusProvider>
);
