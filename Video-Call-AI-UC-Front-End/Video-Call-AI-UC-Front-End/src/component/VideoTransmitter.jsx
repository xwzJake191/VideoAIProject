// VideoTransmitter.jsx
import React, { useContext, useEffect, useRef } from 'react';
import io from 'socket.io-client';
import StatusContext from '../page/StatusContext';

// Assume Flask server is running at localhost:5858
const socket = io('http://localhost:5858');

const VideoTransmitter = () => {
  const { userMediaStream, isTransmitting, socket } = useContext(StatusContext);
  const requestRef = useRef(); // Used to store the return value of requestAnimationFrame

  useEffect(() => {
    console.log(`Effect triggered: isTransmitting=${isTransmitting}, userMediaStream available=${!!userMediaStream}`);

    // If not transmitting, or userMediaStream doesn't exist, or socket is not available, do nothing
    if (!isTransmitting || !userMediaStream || !socket) {
      console.log('Cannot transmit: no userMediaStream found, or not transmitting. Exiting...');
      return;
    }

    const canvas = document.createElement('canvas');
    const video = document.createElement('video');
    video.srcObject = userMediaStream;

    console.log('Created video and canvas elements for transmission.');

    video.onloadedmetadata = () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      console.log(`Video metadata loaded: width=${video.videoWidth}, height=${video.videoHeight}`);
      
      video.play().then(() => {
        console.log('Video playback for transmission started.');

        const sendFrame = () => {
          if (!isTransmitting) {
            console.log('Transmitting has been stopped. Ceasing frame capture.');
            cancelAnimationFrame(requestRef.current); // Cancel requestAnimationFrame loop only when transmission is stopped
            return;
          }
          const context = canvas.getContext('2d');
          context.drawImage(video, 0, 0, canvas.width, canvas.height);
          const imageDataUrl = canvas.toDataURL('image/jpeg');
          console.log('Sending frame...');
          socket.emit('frame', { imageDataUrl });

          requestRef.current = requestAnimationFrame(sendFrame);
        };

        // Start frame transmission loop
        requestRef.current = requestAnimationFrame(sendFrame);
      }).catch(error => {
        console.error('Error playing video for transmission:', error);
      });
    };

    // Cleanup function no longer needs to stop the video stream
    return () => {
      console.log('Cleaning up: Cancelling frame transmission.');
      cancelAnimationFrame(requestRef.current);
    };
  }, [isTransmitting, userMediaStream]);

  // The component doesn't render anything, only responsible for transmission logic
  return null;
};

export default VideoTransmitter;
