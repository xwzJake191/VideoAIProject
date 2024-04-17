import { useState, useEffect } from 'react';

export const useMicrophone = () => {
  const [userAudioStream, setUserAudioStream] = useState(null);
  const [isListening, setIsListening] = useState(false);

  const startMicrophone = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setUserAudioStream(stream);
      setIsListening(true);
    } catch (error) {
      console.error('Error accessing the microphone:', error);
    }
  };

  const stopMicrophone = () => {
    if (userAudioStream) {
      userAudioStream.getTracks().forEach(track => track.stop());
      setUserAudioStream(null);
      setIsListening(false);
    }
  };

  const toggleMicrophone = () => {
    if (isListening) {
      stopMicrophone();
    } else {
      startMicrophone();
    }
  };

  useEffect(() => {
    return () => {
      // Ensure microphone is turned off when the component unmounts
      stopMicrophone();
    };
  }, [userAudioStream]);

  return { userAudioStream, isListening, startMicrophone, stopMicrophone, toggleMicrophone };
};
