import os
import time
import azure.cognitiveservices.speech as speechsdk
import json

def conversation_transcriber_recognition_canceled_cb(evt: speechsdk.SessionEventArgs):
    print('Canceled event')

def conversation_transcriber_session_stopped_cb(evt: speechsdk.SessionEventArgs):
    print('SessionStopped event')

def conversation_transcriber_transcribed_cb(evt: speechsdk.SpeechRecognitionEventArgs):
    print('TRANSCRIBED:')
    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
        # embedding 说话人后单个 string 作为 object 的输出
        # json_text = f"{evt.result.speaker_id}: {evt.result.text}"
        # print(json_text)

        # Json key-value pair 形式输出
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            # 构建一个字典，其中包含说话者 ID 和识别的文本
            transcribed_text = {f"Speaker {evt.result.speaker_id}": evt.result.text}        
            #  json.dumps 将字典转换为 JSON 字符串
            json_text2 = json.dumps(transcribed_text, ensure_ascii=False)
            print(json_text2)
        
        # API 的原始输出
        print('\tText={}'.format(evt.result.text))
        print('\tSpeaker ID={}'.format(evt.result.speaker_id))
    elif evt.result.reason == speechsdk.ResultReason.NoMatch:
        print('\tNOMATCH: Speech could not be TRANSCRIBED: {}'.format(evt.result.no_match_details))

def conversation_transcriber_session_started_cb(evt: speechsdk.SessionEventArgs):
    print('SessionStarted event')

def recognize_from_mic():
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
    speech_config.speech_recognition_language="en-US"

    conversation_transcriber = speechsdk.transcription.ConversationTranscriber(speech_config=speech_config)

    transcribing_stop = False

    def stop_cb(evt: speechsdk.SessionEventArgs):
        #"""callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        nonlocal transcribing_stop
        transcribing_stop = True

    # Connect callbacks to the events fired by the conversation transcriber
    conversation_transcriber.transcribed.connect(conversation_transcriber_transcribed_cb)
    conversation_transcriber.session_started.connect(conversation_transcriber_session_started_cb)
    conversation_transcriber.session_stopped.connect(conversation_transcriber_session_stopped_cb)
    conversation_transcriber.canceled.connect(conversation_transcriber_recognition_canceled_cb)
    # stop transcribing on either session stopped or canceled events
    conversation_transcriber.session_stopped.connect(stop_cb)
    conversation_transcriber.canceled.connect(stop_cb)

    conversation_transcriber.start_transcribing_async()

    # Waits for completion.
    while not transcribing_stop:
        time.sleep(.5)

    conversation_transcriber.stop_transcribing_async()

# Main

try:
    recognize_from_mic()
except Exception as err:
    print("Encountered exception. {}".format(err))