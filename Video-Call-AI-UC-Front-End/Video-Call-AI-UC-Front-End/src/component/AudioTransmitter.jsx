import React, { useContext, useEffect, useRef } from 'react';
import StatusContext from '../page/StatusContext';

const AudioTransmitter = () => {
  const { isRecognizing, userAudioStream, socket } = useContext(StatusContext);
  const mediaRecorderRef = useRef(null);
  const intervalRef = useRef(null); // Used to hold the reference of the timer

  useEffect(() => {
    // Define function to start recording
    const startRecording = () => {
      const options = {
        mimeType: 'audio/webm', // Make sure this is a supported format
        audioBitsPerSecond: 16000
      };
      mediaRecorderRef.current = new MediaRecorder(userAudioStream, options);

      // Handling when data is available
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          const reader = new FileReader();
          reader.onload = () => {
            const audioDataUrl = reader.result;
            console.log('Sending audio data...');
            socket.emit('audio', { audioDataUrl });
          };
          reader.readAsDataURL(event.data);
        }
      };

      mediaRecorderRef.current.start();
      console.log("Audio recording started.");
    };

    // Define function to stop recording
    const stopRecording = () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
        console.log("Audio recording stopped.");
      }
    };

    // Logic to restart recording every 15 seconds
    if (isRecognizing && userAudioStream) {
      startRecording();
      intervalRef.current = setInterval(() => {
        stopRecording();
        startRecording();
      }, 15000);
    } else {
      clearInterval(intervalRef.current);
      stopRecording();
    }

    // Clean up resources when component unmounts
    return () => {
      clearInterval(intervalRef.current);
      stopRecording();
    };
  }, [userAudioStream, isRecognizing, socket]);

  return null;
};

export default AudioTransmitter;
