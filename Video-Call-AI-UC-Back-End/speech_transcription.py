import contextlib
import wave
import os
import shutil
import tempfile
import subprocess
from transformers import pipeline
import sqlite3
import numpy as np
import torchaudio
from pyannote.audio import Audio, Pipeline, Model, Inference
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine

# Initialize transcription pipeline
transcribe_pipe = pipeline("automatic-speech-recognition", model="distil-whisper/distil-large-v3", return_timestamps=True)
speech_emo_classifier = pipeline("audio-classification", model="hughlan1214/Speech_Emotion_Recognition_wav2vec2-large-xlsr-53_240304_SER_fine-tuned2.0")

#####
# Connect to SQLite database and update speaker voice features table
conn = sqlite3.connect('speaker_voice_features.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS speakers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    embedding BLOB
)
''')
conn.commit()

# Initialize speaker diarization and voice embedding extraction model
HUGGINGFACE_ACCESS_TOKEN = "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"
diarization_pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HUGGINGFACE_ACCESS_TOKEN)
embedding_model = Model.from_pretrained("pyannote/embedding", use_auth_token=HUGGINGFACE_ACCESS_TOKEN)
embedding_inference = Inference(embedding_model)
#####

def transcribe_audio(audio_file):
    """Transcribe audio and return the transcript with timestamps"""
    # Request return_timestamps when calling
    transcription_result = transcribe_pipe(audio_file)
    return transcription_result

import wave
import contextlib
import torchaudio
from pyannote.core import Segment

def diarization_and_embeddings(audio_file, output_dir):
    """
    Run speaker diarization model, extract voice embeddings, and save RTTM file.
    
    Args:
    audio_file (str): Path to the audio file.
    model: Initialized speaker diarization model.
    output_dir (str): Output directory path.
    
    Returns:
    dict: Dictionary containing voice embeddings for each speaker.
    """
    # Get audio duration to aviod "NoneType.__format__" error 
    with contextlib.closing(wave.open(audio_file, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        audio_duration = frames / float(rate)
    print(f"Audio file {audio_file} duration: {audio_duration} seconds")

    # Use pretrained model to run speaker diarization and get turn information for each speaker
    diarization = diarization_pipe(audio_file)
    
    # Initialize dictionaries: embeddings to store voice embeddings for each speaker, time_segments to store time segments for each speaker
    embeddings = {}
    time_segments = {}
    diarazation_results = []

    # Iterate over each turn in the diarization result (i.e., each speaker's speech)
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # Ensure the segment is within the duration of the audio file
        start = max(0, turn.start)
        end = min(turn.end, audio_duration)

        # Skip the processing if the duration is zero or negative
        if end <= start:
            print(f"Skipping segment from {start} to {end} due to invalid duration.")
            continue

        # Create a Segment object based on the current turn, which will be used for voice embedding extraction
        excerpt = Segment(start, end)
        print(f"Processing excerpt from {start} to {end} seconds.")

        # Call the voice embedding extraction model to extract voice embeddings for the current turn
        embedding = embedding_inference.crop(audio_file, excerpt).data
        print(embedding)
        print(type(embedding))

        try:
            # If the current speaker is not in the embeddings dictionary, initialize empty lists to store voice embeddings and time segments for that speaker
            if speaker not in embeddings:
                embeddings[speaker] = []
                time_segments[speaker] = []

            # Append the voice embedding extracted for the current turn to the list of voice embeddings for that speaker
            embeddings[speaker].append(embedding)

            # Append the time segment information for the current turn to the list of time segments for that speaker, recording the start and end times
            time_segments[speaker].append((start, end))
        except Exception as e:
            print(f"Error processing segment {start}-{end}: {str(e)}")

    # Use clustering and database comparison method to identify different speakers and get speaker names
    speaker_names = add_or_identify_speakers(embeddings)

    # Build the output RTTM file path
    output_file = os.path.join(output_dir, os.path.basename(audio_file).replace('.wav', '.rttm'))

    # Open the RTTM file and write the diarization results
    with open(output_file, "w") as file:
        for speaker, segments in time_segments.items():
            for start, end in segments:
                duration = end - start
                label = speaker_names[speaker]
                file.write(f"SPEAKER {output_file} 1 {start} {duration} <NA> <NA> {label} <NA> <NA>\n")
                diarazation_results.append({"start": start, "end": end, "speaker": label})

    # Print information about the saved results
    print(f"Diarization results saved to {output_file}")
    return diarazation_results

def add_or_identify_speakers(embeddings):
    """Identify speakers by comparing voice embeddings with those in the database, or add new ones."""
    X = np.vstack([embedding.flatten() for speaker in embeddings.values() for embedding in speaker])

    # Initialize the speaker names dictionary
    speaker_names = {}

    # If only one sample, handle it directly without clustering
    if X.shape[0] == 1:
        embedding_blob = X.flatten().tobytes()
        # Search for the closest match in the database
        cursor.execute("SELECT id, name, embedding FROM speakers")
        best_match = None
        min_distance = float('inf')
        
        for row in cursor.fetchall():
            existing_embedding = np.frombuffer(row[2], dtype=np.float32).reshape(-1)
            distance = cosine(X.flatten(), existing_embedding)
            if distance < min_distance:
                min_distance = distance
                best_match = row

        # Determine if a new speaker should be added or an existing one identified
        if best_match and min_distance < 0.1:
            speaker_name = best_match[1]  # Match found, use existing speaker name
        else:
            # No match found, add new speaker to the database
            cursor.execute("INSERT INTO speakers (embedding) VALUES (?)", (embedding_blob,))
            conn.commit()
            new_id = cursor.lastrowid
            speaker_name = f"Speaker{new_id}"
            cursor.execute("UPDATE speakers SET name = ? WHERE id = ?", (speaker_name, new_id))
            conn.commit()

        # Assign the determined speaker name to the only speaker in embeddings
        speaker_names[list(embeddings.keys())[0]] = speaker_name
        return speaker_names

    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.1, linkage='average')
    labels = clustering.fit_predict(X)

    # Map clustered labels to speaker embeddings
    unique_speakers = np.unique(labels)
    speakers_list = list(embeddings.keys())

    for speaker_label in unique_speakers:
        indices = np.where(labels == speaker_label)[0]
        speaker_embeddings = [X[index] for index in indices]
        mean_embedding = np.mean(speaker_embeddings, axis=0)

        # Convert to binary format for database comparison
        embedding_blob = mean_embedding.tobytes()

        # Search for the closest match in the database
        cursor.execute("SELECT id, name, embedding FROM speakers")
        best_match = None
        min_distance = float('inf')
        
        for row in cursor.fetchall():
            existing_embedding = np.frombuffer(row[2], dtype=np.float32)
            distance = cosine(mean_embedding, existing_embedding)
            if distance < min_distance:
                min_distance = distance
                best_match = row

        if best_match and min_distance < 0.1:
            speaker_name = best_match[1]
        else:
            cursor.execute("INSERT INTO speakers (embedding) VALUES (?)", (embedding_blob,))
            conn.commit()
            new_id = cursor.lastrowid
            speaker_name = f"Speaker{new_id}"
            cursor.execute("UPDATE speakers SET name = ? WHERE id = ?", (speaker_name, new_id))
            conn.commit()

        for idx in indices:
            speaker_names[speakers_list[idx]] = speaker_name

    return speaker_names

def slice_audio(audio_file, start_time, end_time, output_file):
    """Slice an audio segment from the original audio file based on given start and end timestamps"""
    command = [
        'ffmpeg',
        '-y', 
        '-i', audio_file,
        '-ss', str(start_time),
        '-to', str(end_time),
        '-acodec', 'copy', 
        output_file
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def classify_speech_emotions(audio_file):
    """
    Classifies speech emotions in an audio file and returns the top three emotions with their scores formatted as percentages.

    Args:
    audio_path (str): The path to the audio file to be classified.

    Returns:
    dict: A dictionary containing the top three emotions and their corresponding scores in percentage.
    """

    # Perform classification
    result = speech_emo_classifier(audio_file)

    # Extract the top 2 emotion labels
    top_three = result[:2]

    # Create the new format to display the results
    formatted_emo_result = {}
    for item in top_three:
        label = item['label']
        score = round(item['score'] * 100, 2)  # Convert the score to percentage with two decimal places
        formatted_emo_result[label] = f"{score}%"

    return formatted_emo_result

# Need modify, slice with diarazation duration, then fit word to diarazation duration according to timestamp
def format_transcription(transcription_result, audio_file):
    """Format the transcription result into a dictionary for easy processing, and analyze emotions"""
    segments = []
    temp_dir = tempfile.mkdtemp()  # Create a temporary directory to store sliced audio files

    try:
        # Loop through each segment in the transcription result
        for segment in transcription_result['chunks']:
            # Generate a unique audio file name based on timestamps
            start_ms = int(segment['timestamp'][0] * 1000)  # Start time in milliseconds
            end_ms = int(segment['timestamp'][1] * 1000)    # End time in milliseconds
            segment_audio = f"{temp_dir}/{start_ms}_{end_ms}.webm"
            
            # Call slice_audio function to slice the original audio file based on timestamps
            slice_audio(audio_file, segment['timestamp'][0], segment['timestamp'][1], segment_audio)# Need modify, slice with diarazation duration
            
            # Analyze emotions
            emotions = classify_speech_emotions(segment_audio)
            
            # Save emotion analysis results and transcription text to segments list
            segments.append({
                "start": segment['timestamp'][0],
                "end": segment['timestamp'][1],
                "text": segment['text'],
                "emotion": emotions
            })

    finally:
        shutil.rmtree(temp_dir)  # Clean up temporary directory

    return segments

# # # Example usage
# audio_file_path = 'speech_transcription.py'
# transcription_result = transcribe_audio(audio_file_path)
# # formatted_segments = format_transcription(transcription_result, audio_file_path)
# # print(formatted_segments)
# print(transcription_result)


# # Test audio file
# audio_file = "/Users/Hugh/Downloads/Flask-test-audio/audio_2024-04-15_08-29-11-217813.wav"
# output_dir = "/Users/Hugh/Downloads/Flask-test-audio"

# results = diarization_and_embeddings(audio_file, output_dir)
# for result in results:
#     print(result)