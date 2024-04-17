import React, { useContext }  from 'react';
import './index.css';
import { useCamera } from '../hook/useCamera';
import CameraDisplay from '../component/CameraDisplay.jsx';
import { useVideoTransmitter } from '../hook/sendVideoStream.js';
import VideoTransmitter from '../component/VideoTransmitter.jsx';
import StatusContext from './StatusContext';
import { sendAudioStream } from '../hook/sendAudioStream.js';
import AudioTransmitter from '../component/AudioTransmitter.jsx';



export default function Main() {
  const { toggleCamera, userMediaStream, isTransmitting, toggleTransmitting, toggleRecognizing, isRecognizing, toggleMicrophone } = useContext(StatusContext);
  
  return (
    <div className='main-container'>
    <div className='frame'>
      <div className='frame-1'>
        <div className='frame-2'>
          <div className='frame-3'>
            <div className='protect' />
          </div>
          <div className='frame-4'>
            <div className='grid-view'>
              <div className='cta'>
                <div className='button'>
                  <div className='icons'>
                    <div className='vector' />
                  </div>
                </div>
              </div>
              <div className='line' />
            </div>
            <div className='grid-view-5'>
              <div className='cta-6'>
                <div className='button-7'>
                  <div className='icons-8'>
                    <div className='vector-9' />
                  </div>
                </div>
              </div>
              <div className='line-a' />
            </div>
            <div className='grid-view-b'>
              <div className='cta-c'>
                <div className='button-d'>
                  <div className='icons-e'>
                    <div className='vector-f' />
                  </div>
                </div>
              </div>
              <div className='line-10' />
            </div>
            <div className='grid-view-11'>
              <div className='cta-12'>
                <div className='button-13'>
                  <div className='icons-14'>
                    <div className='vector-15' />
                  </div>
                </div>
              </div>
              <div className='line-16' />
            </div>
          </div>
          <div className='frame-17'>
            <button className='microphone'>
              <div className='button-18'>
                <div className='component' />
                <span className='time-stamp'>13:03:34</span>
              </div>
            </button>
          </div>
        </div>
        <div className='canvas' />
        <CameraDisplay userMediaStream={userMediaStream} />
        <div className='frame-19'>
          <div className='frame-1a'>
            <button className='volume'>
              <div className='button-1b'>
                <div className='speaker'>
                  <div className='vector-1c' />
                </div>
                <div className='frame-1d'>
                  <div className='bar'>
                    <div className='bar-1e' />
                  </div>
                  <div className='rectangle' />
                </div>
              </div>
            </button>
          </div>
          <div className='frame-1f'>
            <div className='frame-20'>
              <button className='Mircophone' onClick={toggleMicrophone}>
                <div className='button-21'>
                  <div className='icons-22'>
                    <div className='icons-microphone'>
                      <div className='vector-23' />
                    </div>
                    <div className='component-24' />
                  </div>
                  <div className='icons-collapse-arrow'>
                    <div className='vector-25' />
                  </div>
                </div>
              </button>
              <button className='Camera' onClick={toggleCamera}>
                <div className='button-26'>
                  <div className='icons-27'>
                    <div className='vector-28' />
                  </div>
                  <div className='icons-collapse-arrow-29'>
                    <div className='vector-2a' />
                  </div>
                </div>
              </button>
              <div className='cta-2b'>
                <div className='button-2c'>
                  <div className='icons-2d'>
                    <div className='vector-2e' />
                  </div>
                  <div className='icons-collapse-arrow-2f'>
                    <div className='vector-30' />
                  </div>
                </div>
              </div>
              <button className='cta-31'>
                <div className='button-32'>
                  <div className='icons-33'>
                    <div className='vector-34' />
                  </div>
                  <div className='icons-collapse-arrow-35'>
                    <div className='vector-36' />
                  </div>
                </div>
              </button>
              <div className='cta-37'>
                <div className='button-38'>
                  <div className='icons-39'>
                    <div className='vector-3a' />
                  </div>
                </div>
              </div>
              <div className='line-3b' />
              <button className='transcribe' onClick={toggleRecognizing}>
                <div className='button-3d'>
                  <div className='icons-3e'>
                    <div className='vector-3f' />
                  </div>
                  <div className='icons-collapse-arrow-40'>
                    <div className='vector-41' />
                  </div>
                </div>
              </button>
              <AudioTransmitter isRecognizing={isRecognizing} />
              <button className='infer' onClick={toggleTransmitting}>
                <div className='button-42'>
                  <div className='icons-43'>
                    <div className='vector-44' />
                  </div>
                  <div className='icons-collapse-arrow-45'>
                    <div className='vector-46' />
                  </div>
                </div>
              </button>
              <VideoTransmitter isTransmitting={isTransmitting} />
            </div>
          </div>
          <div className='empty-container'>
            <button className='stop'>
              <div className='button-47'>
                <span className='leave-meeting'>Leave Meeting</span>
              </div>
            </button>
          </div>
        </div>
      </div>
      <div className='rightbar'>
        <div className='sidebar'>
          <div className='tab-header'>
            <div className='component-48'>
              <button className='bg' />
              <div className='button-49'>
                <div className='button-4a'>
                  <div className='icons-4b'>
                    <div className='vector-4c' />
                  </div>
                  <span className='leave-meeting-4d'>Topic Prompt</span>
                </div>
              </div>
              <div className='button-4e'>
                <div className='button-4f'>
                  <div className='icons-50'>
                    <div className='vector-51' />
                  </div>
                  <span className='leave-meeting-52'>Topic Segment</span>
                </div>
              </div>
            </div>
          </div>
          <div className='text-area'>
            <div className='frame-53' />
          </div>
        </div>
      </div>
      <div className='rightbar-54'>
        <div className='sidebar-55'>
          <div className='tab-header-56'>
            <div className='component-57'>
              <button className='bg-58' />
              <div className='button-59'>
                <div className='button-5a'>
                  <div className='icons-5b'>
                    <div className='vector-5c' />
                  </div>
                  <span className='leave-meeting-5d'>Topic Prompt</span>
                </div>
              </div>
              <div className='button-5e'>
                <div className='button-5f'>
                  <div className='icons-60'>
                    <div className='vector-61' />
                  </div>
                  <span className='leave-meeting-62'>Topic Segment</span>
                </div>
              </div>
            </div>
          </div>
          <div className='text-area-63'>
            <div className='frame-64' />
          </div>
        </div>
      </div>
    </div>
  </div>
  );
}
