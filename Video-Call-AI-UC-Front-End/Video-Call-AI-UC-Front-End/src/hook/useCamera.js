// useCamera.js
import { useState, useEffect } from 'react';

export const useCamera = () => {
  const [userMediaStream, setUserMediaStream] = useState(null);
  const [isCameraOn, setIsCameraOn] = useState(false);

  const startCamera = async () => {
    try {
      const constraints = {
        video: {
          width: { ideal: 1280 }, 
          height: { ideal: 720 }, 
          // Set camera resolution
        }
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      setUserMediaStream(stream);
      setIsCameraOn(true);
    } catch (error) {
      console.error('Error accessing the camera:', error);
    }
  };
  

  const stopCamera = () => {
    if (userMediaStream) {
      userMediaStream.getTracks().forEach(track => track.stop());
      setUserMediaStream(null);
      setIsCameraOn(false);
    }
  };

  const toggleCamera = () => {
    if (isCameraOn) {
      stopCamera();
    } else {
      startCamera();
    }
  };

  useEffect(() => {
    return () => {
      if (userMediaStream) {
        stopCamera();
      }
    };
  }, [userMediaStream]);

  return { toggleCamera, isCameraOn, userMediaStream };
};
