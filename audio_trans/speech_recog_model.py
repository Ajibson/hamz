import re
import os
import json
import torch
import torchaudio
import transformers
import numpy
# from pathlib import Path
from transformers import AutoModelForCTC, Wav2Vec2Processor
from pydub import AudioSegment

# The model was finetuned  on facebook/wav2vec2 model and the weights was pushed to the transformer-hub(my repository on my transformer hub account).
#Link to the training is on github( my github account).. : https://github.com/AbooMardiiyah/Hausa-speech-recognition-xformers/blob/main/Hausa_speech_recog_Transformers.ipynb


#Loading the processor and the model from my transformer hub account...
processor = Wav2Vec2Processor.from_pretrained("Tiamz/hausa-4-ha-wa2vec-data-aug-xls-r-300m")
model = AutoModelForCTC.from_pretrained("Tiamz/hausa-4-ha-wa2vec-data-aug-xls-r-300m")

def convert_to_wav(filename):
     """Takes an audio file of non .wav format and converts to .wav"""
     # Import audio file
     audio = AudioSegment.from_file(filename)
  
     # Create new filename
     new_filename = filename.split(".")[0] + ".wav"
  
     # Export file as .wav
     audio.export(new_filename, format='wav')
     print(f"Converting {filename} to {new_filename}...")


def show_pydub_stats(filename):
    """Returns different audio attributes related to an audio file."""
    # Create AudioSegment instance
    audio_segment = AudioSegment.from_file(filename)
  
    # Print audio attributes and return AudioSegment instance
    print(f"Channels: {audio_segment.channels}")
    print(f"Sample width: {audio_segment.sample_width}")
    print(f"Frame rate (sample rate): {audio_segment.frame_rate}")
    print(f"Frame width: {audio_segment.frame_width}")
    print(f"Length (ms): {len(audio_segment)}")
    return audio_segment

def get_transcription(filename):
    res={}
    #load audio files
    speech,sr=torchaudio.load(filename)
  
    speech=speech.squeeze()
    resampler=torchaudio.transforms.Resample(sr,16000)
    speech=resampler(speech)
    # tokenize our wav
    input_values=processor(speech, return_tensors="pt",sampling_rate=16000)["input_values"]
    #perform inference
    logits=model(input_values)["logits"]
    #use argmax to predict
    predicted_ids=torch.argmax(logits,dim=-1)
    #decode Ids to text
    res["guess"]=processor.decode(predicted_ids[0])
    res["truth"]=processor.decode(predicted_ids[0],skip_special_tokens=True)
    return res
  


