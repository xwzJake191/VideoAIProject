import React, { useContext, useEffect } from 'react';
import StatusContext from '../src/page/StatusContext';
import axios from 'axios';
import * as speechsdk from 'microsoft-cognitiveservices-speech-sdk';

const SpeechRecognition = () => {
    const { isRecognizing } = useContext(StatusContext);

    useEffect(() => {
        let recognizer;

        const fetchSpeechTokenAndSetupRecognizer = async () => {
            if (isRecognizing) {
                try {
                    const response = await axios.get('http://localhost:3001/api/get-speech-token');
                    const { token, region } = response.data;
                    const speechConfig = speechsdk.SpeechConfig.fromAuthorizationToken(token, region);
                    console.log(`Managed to fetch token`);
                    speechConfig.speechRecognitionLanguage = 'en-US';

                    // Enable audio and transcription logging
                    speechConfig.enableAudioLogging();

                    // Create a new recognizer instance for conversation transcription
                    const audioConfig = speechsdk.AudioConfig.fromDefaultMicrophoneInput();
                    recognizer = new speechsdk.ConversationTranscriber(speechConfig, audioConfig);
                    
                    // Set transcription event handling functions
                    recognizer.transcribing = (s, e) => {
                        console.log(`Transcribing: Text=${e.result.text}`);
                    };

                    recognizer.transcribed = (s, e) => {
                        console.log(`Test test, see if pauses or speaker changes are detected.`);
                        console.log(`Transcribed: Text=${e.result.text} Speaker ID=${e.result.speakerId}`);
                    }; // Later on save transcribed text and transmit to Flask, detect pauses

                    recognizer.sessionStarted = (s, e) => {
                        console.log("Session started event");
                    };

                    recognizer.sessionStopped = (s, e) => {
                        console.log("Session stopped event");
                        recognizer.stopTranscribingAsync();
                    };

                    recognizer.canceled = (s, e) => {
                        console.log(`Canceled: Reason=${e.reason}`);
                        recognizer.stopTranscribingAsync();
                    };

                    // Start transcription
                    recognizer.startTranscribingAsync();
                    console.log(`Speak to mic`);

                } catch (error) {
                    console.error("Error fetching speech service token:", error);
                }
            } else {
                if (recognizer) {
                    recognizer.stopTranscribingAsync(() => {
                        recognizer.close();
                        recognizer = null;
                    });
                }
            }
        };

        fetchSpeechTokenAndSetupRecognizer();

        return () => {
            if (recognizer) {
                recognizer.stopTranscribingAsync(() => {
                    recognizer.close();
                });
            }
        };
    }, [isRecognizing]);

    return null;
};

export default SpeechRecognition;
