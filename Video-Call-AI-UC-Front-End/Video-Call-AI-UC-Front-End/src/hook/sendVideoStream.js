// sendVideoStream.js
import { useState } from 'react';

export const useVideoTransmitter = () => {
  const [isTransmitting, setIsTransmitting] = useState(false);

  const toggleTransmitting = () => setIsTransmitting(prev => !prev);

  return { isTransmitting, toggleTransmitting };
};