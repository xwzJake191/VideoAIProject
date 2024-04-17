# model_loader.py
import torch
from torchvision import transforms
from transformers import pipeline
from image_process import ResNet50, LSTMPyTorch

class ModelLoader:
    def __init__(self):
        # 初始化所有模型
        self.load_models()

    def load_models(self):
        """加载并初始化所有模型"""
        self.pth_backbone_model = ResNet50(7, channels=3)
        self.pth_backbone_model.load_state_dict(torch.load('FER_static_ResNet50_AffectNet.pt', map_location=torch.device('cpu')))
        self.pth_backbone_model.eval()

        self.pth_LSTM_model = LSTMPyTorch()
        self.pth_LSTM_model.load_state_dict(torch.load('FER_dinamic_LSTM_Aff-Wild2.pt', map_location=torch.device('cpu')))
        self.pth_LSTM_model.eval()

        self.labels = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust', 6: 'Anger'}

        self.speech_emo_classifier = pipeline("audio-classification", model="hughlan1214/Speech_Emotion_Recognition_wav2vec2-large-xlsr-53_240304_SER_fine-tuned2.0")

        self.transcribe_pipe = pipeline("automatic-speech-recognition", model="distil-whisper/distil-large-v3", return_timestamps=True)

    def get_backbone_model(self):
        return self.pth_backbone_model

    def get_lstm_model(self):
        return self.pth_LSTM_model
    
    def get_labels(self):
        return self.labels

    def get_speech_emotion_classifier(self):
        return self.speech_emo_classifier
    
    def get_transcribe_pipe(self):
        return self.transcribe_pipe
