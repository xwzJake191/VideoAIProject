import React, { useState, useEffect } from 'react';
import StatusContext from '../page/StatusContext.js'; 
import { useCamera } from '../hook/useCamera.js';
import { useVideoTransmitter } from '../hook/sendVideoStream.js';
import { useMicrophone } from '../hook/useMicrophone.js';
import { sendAudioStream } from '../hook/sendAudioStream.js'; 
import io from 'socket.io-client'; // Import Socket.IO client

const StatusProvider = ({ children }) => {
  const { userMediaStream, isCameraOn, toggleCamera } = useCamera();
  const { isTransmitting, toggleTransmitting } = useVideoTransmitter();
  const { toggleMicrophone, isListening, userAudioStream } = useMicrophone();
  const { isRecognizing, toggleRecognizing } = sendAudioStream(); // Initialize sendAudioStream hook
  const [processedImageDataUrl, setProcessedImageDataUrl] = useState('');
  const [socket, setSocket] = useState(null); // State to hold the Socket.IO instance

  // Create Socket.IO instance
  useEffect(() => {
    const newSocket = io('http://localhost:5858');
    setSocket(newSocket);
    console.log('Socket.IO connected');

    return () => newSocket.close(); // Cleanup function
  }, []);

  return (
    <StatusContext.Provider value={{
      isCameraOn,
      userMediaStream,
      isTransmitting,
      toggleCamera,
      toggleTransmitting,
      processedImageDataUrl,
      setProcessedImageDataUrl,
      isListening,
      toggleMicrophone,
      userAudioStream,
      isRecognizing,
      toggleRecognizing, // Add new state functions to the context
      socket
    }}>
      {children}
    </StatusContext.Provider>
  );
};

export default StatusProvider;
