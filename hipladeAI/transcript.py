import whisper
from openai import OpenAI
import PyPDF2
import cv2
import matplotlib.pyplot as plt
import moviepy.editor as mp
import soundfile as sf
import librosa
import numpy as np
from fpdf import FPDF

def mp4_to_mp3(file,cut=1e-5):
    video = mp.VideoFileClip(file)
    audio = video.audio
    audio.write_audiofile(file.replace("mp4","mp3"))
    y, sr = librosa.load(file.replace("mp4","mp3"), sr=None) 
    y = y[np.abs(y)>cut]
    chunk_duration = 30
    samples = int(chunk_duration * sr)
    chunks = [y[i:i + samples] for i in range(0, len(y), samples)]
    files = []
    for i, chunk in enumerate(chunks):
        output_file = f"audio{i}.mp3"
        sf.write(output_file, chunk, sr)
        files.append(output_file)
    return np.array(files)

def transcribeMP4(file,modelType="small"):
    fileList = mp4_to_mp3(file)
    text = ""
    for f in fileList:
        model = whisper.load_model(modelType)
        text = text + model.transcribe(f,no_speech_threshold=0.75, language="pt")["text"]
        #print(text)
    return text

def processPDF(file='OSLER - CLINICA MÉDICA - AMBULATÓRIO revisado AQ.docx.pdf'):
    reader = PyPDF2.PdfReader(file)
    text = ""
    goal = ""
    evaluation = ""
    for p in reader.pages:
        text = p.extract_text()
        if "objetivo" in text.lower():
            goal = text
        if "correto" in text.lower():
            evaluation = evaluation + text
    evaluation = evaluation.replace("\n", " ")
    goal = goal.replace("\n", " ")
    return goal, evaluation

def writeManuscript(text):
    output=FPDF(format='A4',unit='mm')
    output.add_page()
    output.set_font("Arial", "B",20)
    output.write(20,"Avaliação:\n")
    output.set_font("Arial", "",12)
    output.write(12,text)
    output.output('relatorio.pdf','F')

def smartEval(fileMP4,pdfFile,api_key):
    client = OpenAI(api_key=api_key)
    transcribed = transcribeMP4(fileMP4)
    goal, evaluation = processPDF(pdfFile)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Você é um professor da área médica"},
            {"role": "system", "content": "Você deverá avaliar a prova com o seguinte objetivo:"+goal},
            {"role": "system", "content": "Você deverá avaliar os seguintes critérios:"+evaluation},
            {"role": "user", "content": "Avalie a prova do aluno, que foi transcrita de um áudio, desconsiderando falhas na transcrição do áudio:"+transcribed}]
    )
    result = response.choices[0].message.content
    writeManuscript(result)
    return result
    
if __name__ == '__main__':
    key = 'Please provide a key for chatgpt'
    mp4 = "Sala 3 _ OSCE UnP-20221123_155948-Gravação de Reunião.mp4"
    pdf = "OSLER - CLINICA MÉDICA - AMBULATÓRIO revisado AQ.docx.pdf"
    smartEval(mp4,pdf,key)
