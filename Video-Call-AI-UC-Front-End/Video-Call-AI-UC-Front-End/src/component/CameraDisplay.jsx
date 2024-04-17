import React, { useRef, useEffect, useContext } from 'react';
import StatusContext from '../page/StatusContext';

const CameraDisplay = () => {
  const videoRef = useRef(null);
  // Destructure the required states and socket instance from StatusContext
  const { userMediaStream, processedImageDataUrl, setProcessedImageDataUrl, socket } = useContext(StatusContext);

  useEffect(() => {
    console.log('CameraDisplay component mounted.');

    if (!socket) return; // Ensure socket instance exists

    // Listen for processed image data
    socket.on('frame_response', (data) => {
      console.log('Received processed frame from backend:', data.imageData);
      if (data && data.imageData) {
        setProcessedImageDataUrl(data.imageData); // Update processed image URL
        console.log('Updating processed image data URL.');

        // Send acknowledgment back to the server
        console.log('Sending acknowledgment back to server.');
        socket.emit('frame_received', { message: 'Frame received successfully.' });
      }
    });

    // Remove listeners when component unmounts
    return () => {
      console.log('Cleaning up: Removing socket event listeners.');
      socket.off('connect');
      socket.off('disconnect');
      socket.off('test_message');
      socket.off('frame_response');
    };
  }, [socket, setProcessedImageDataUrl]);

  useEffect(() => {
    // Display live camera feed if there's no processed image data
    if (videoRef.current && userMediaStream && !processedImageDataUrl) {
      console.log('Displaying live camera feed.');
      videoRef.current.srcObject = userMediaStream;
    }
  }, [userMediaStream, processedImageDataUrl]);

  return (
    <div>
      {processedImageDataUrl ? (
        <img src={`data:image/jpeg;base64,${processedImageDataUrl}`} alt="Processed Frame" />
      ) : (
        userMediaStream && <video ref={videoRef} autoPlay playsInline />
      )}
    </div>
  );
};

export default CameraDisplay;
