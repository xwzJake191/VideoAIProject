import React from 'react';

const StatusContext = React.createContext({
  // Existing states and functions
  isCameraOn: false,
  userMediaStream: null, // Media stream from camera
  isTransmitting: false,
  processedImageDataUrl: '',
  toggleCamera: () => {}, // Function to control camera toggle
  toggleTransmitting: () => {},
  setProcessedImageDataUrl: () => {},
  socket: null, // Existing socket state

  // New states and functions for speech recognition
  isRecognizing: false, // Whether speech recognition is in progress
  setIsRecognizing: () => {}, 
  toggleRecognizing: () => {}, // Function to toggle speech recognition state

  // New states and functions for microphone
  isListening: false, // Whether microphone is actively listening
  toggleMicrophone: () => {}, // Function to toggle microphone state
  userAudioStream: null, // Audio stream from microphone
});

export default StatusContext;
