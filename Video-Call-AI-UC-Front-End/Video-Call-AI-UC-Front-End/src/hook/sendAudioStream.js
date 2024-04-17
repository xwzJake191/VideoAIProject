import { useState, useCallback, useEffect } from 'react';

export const sendAudioStream = () => {
  const [isRecognizing, setIsRecognizing] = useState(false);

  // Use useCallback to ensure toggleRecognizing function is not recreated on every component re-render
  const toggleRecognizing = useCallback(() => {
    console.log('Attempting to toggle recognition state.'); // Log the attempt to toggle state
    setIsRecognizing(prevIsRecognizing => {
      console.log(`Current state before toggle: ${prevIsRecognizing}`); // Log the state before toggle
      return !prevIsRecognizing;
    });
  }, []);

  // Listen for changes in isRecognizing state and log them
  useEffect(() => {
    console.log(`Toggling speech recognition. Current state: ${isRecognizing}`);
  }, [isRecognizing]); 

  return { isRecognizing, setIsRecognizing, toggleRecognizing };
};
