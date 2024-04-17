from flask import Flask, request, jsonify, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import argparse
import logging
import logging.config
import conf
import cv2
import numpy as np
import base64
import os
from io import BytesIO
from datetime import datetime
from image_process import predict_and_draw_opencv
from speech_transcription import transcribe_audio, slice_audio, format_transcription, classify_speech_emotions
from model_loader import ModelLoader  

logging.config.dictConfig(conf.dictConfig)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# 创建模型加载器实例
model_loader = ModelLoader()

# 使用模型加载器获取模型实例
pth_backbone_model = model_loader.get_backbone_model()
pth_LSTM_model = model_loader.get_lstm_model()
DICT_EMO = model_loader.get_labels

# Initialize pipelines
speech_emo_classifier = model_loader.get_speech_emotion_classifier()
transcribe_pipe = model_loader.get_transcribe_pipe

processed_frames_count = 0


@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@app.route('/')
def index():
    return "Flask server is running"

@socketio.on('frame')
def handle_frame(data):
    try:
        print('Received frame from frontend')
        image_data_url = data.get('imageDataUrl')
        if not image_data_url:
            logger.error('No image data URL provided.')
            emit('error', {'message': 'No image data URL provided.'})
            return

        base64_str = image_data_url.split(",")[1]
        image_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.error('Failed to decode image.')
            emit('error', {'message': 'Failed to decode image.'})
            return

        processed_img = predict_and_draw_opencv(
            img,
            pth_backbone_model,
            pth_LSTM_model,
            DICT_EMO,
            max_num_faces=1,
            min_detection_confidence=0.5,
            refine_landmarks=False,
            min_tracking_confidence=0.5
        )

        global processed_frames_count
        processed_frames_count += 1
        print(f"Successfully processed {processed_frames_count} frames")

        _, buffer = cv2.imencode('.jpeg', processed_img)
        imageDataUrl = base64.b64encode(buffer).decode('utf-8')

        print(f"Trying to send {processed_frames_count} frames")
        emit('frame_response', {'imageData': imageDataUrl})
        print(f"Sent {processed_frames_count} frames")
    except Exception as e:
        logger.error(f'Error processing frame: {e}')
        emit('error', {'message': 'Error processing frame.'})


@socketio.on('audio')
def handle_audio(data):
    try:
        print("Received audio data from frontend")
        audio_data_url = data.get('audioDataUrl')
        if not audio_data_url:
            emit('error', {'message': 'No audio data URL provided.'})
            return

        base64_str = audio_data_url.split(",")[1]
        audio_data = base64.b64decode(base64_str)

        if len(audio_data) == 0:
            emit('error', {'message': 'Received empty audio data.'})
            return

        # Save the audio file and return the file path
        file_path = save_audio_to_file(audio_data, '/Users/Hugh/Downloads/Flask-test-audio', 'wav') #Saving file could not be loaded by torchaudio, need further address

        # Process the audio data using the file path
        transcription_result = transcribe_audio(file_path)
        formatted_segments = format_transcription(transcription_result, file_path)
        print(formatted_segments)

        emit('transcription', {'status': 'Audio received and processed successfully', 'transcription': formatted_segments})
    except Exception as e:
        logger.error(f'Error processing audio data: {e}')
        emit('error', {'message': f'Error processing audio data: {str(e)}'})

def save_audio_to_file(audio_data, save_directory, file_extension):
    """
    Saves audio data from a buffer to a file in the specified directory with the specified extension. Optional, might be removed after debugging
    """
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    filename = datetime.now().strftime(f"audio_%Y-%m-%d_%H-%M-%S-%f.{file_extension}")
    file_path = os.path.join(save_directory, filename)
    with open(file_path, 'wb') as f:
        f.write(audio_data)  
    print(f"Audio data saved successfully to {file_path}")
    return file_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--port',type=int,default=5858, help="Running port")
    parser.add_argument("-H","--host",type=str,default='0.0.0.0', help="Address to broadcast")
    args = parser.parse_args()
    logger.debug("Starting Flask server with Socket.IO")
    socketio.run(app, host=args.host, port=args.port, debug=True)
