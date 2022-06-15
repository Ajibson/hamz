from turtle import up
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
import os
from audio_trans.speech_recog_model import convert_to_wav,get_transcription
import jiwer

@api_view(['GET', "POST"])
def record(request):
    text = []
    path = os.path.abspath(os.path.dirname(__file__))
    MAX_FILE_SIZE = 25 * 1024 * 1024
    UPLOAD_EXTENSIONS = ['mp3', 'mp4', 'wav','jpeg']
    if request.method == 'POST' and 'file' in request.FILES:
        uploaded_file = request.data.get('file')
        filename = uploaded_file.name
        file_ext = filename.split('.')[1]
        if uploaded_file.size < MAX_FILE_SIZE:
            if file_ext in UPLOAD_EXTENSIONS:
                #get file from client with request and save the file
                try:
                    file_path = os.mkdir(os.path.join(path,'data_files'))
                except:
                    pass
                with open(os.path.join(path,f'data_files/{filename}'), "wb") as file:
                    file.write(uploaded_file.file.read())

                for root, dirs, files in os.walk(os.path.join(os.path.abspath(os.path.dirname(__file__)),'data_files')):
                    for f in files:
                        if f.endswith('.wav') == False:
                            print('#### Ready to convert file to .wav ######')
                            convert_to_wav(os.path.join(path,'data_files', f))
                            audio_wav = os.path.join(path,'data_files', request.files.get('file').filename.split('.')[0]+'.wav')
                            result = get_transcription(audio_wav)
                            trans_text=result['guess']
                            # print(trans_text)
                            wer_res=round(jiwer.wer(references=result['truth'],predictions=result['guess']),2)
                            text.append(trans_text,wer_res)
                        else:
                            return Response({'result': 'Incomplete', 'message': 'file not yet .wav format'}, status=status.HTTP_417_EXPECTATION_FAILED)
                
                return Response({"text": str(text[0]), 'word_error_rate':int(text[1])})
            
            else:
                return Response({"result":"failure", "error":f"Invalid file type. allowed extentions are {','.join(UPLOAD_EXTENSIONS)}"}, status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
        else:
            return Response({"result":"failure", "error":"max file size exceeded. min is 25MB"}, status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
    
        
    
        
    