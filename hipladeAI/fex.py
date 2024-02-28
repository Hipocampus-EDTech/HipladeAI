import cv2
import face_recognition
import numpy as np
from feat import Detector
import pandas as pd
import os
import time



def fex(video_path, user_encodings, user_name):
    start = time.time()
    user_found = ''

    white_image = np.ones((512, 512, 3), dtype=np.uint8) * 255
    try:
        #Primeiro passo é salvar as codificações na variavel
        def load_encodings(encodings_file):
            with open(encodings_file, 'r') as file:
                encodings = [[float(num) for num in line.split(',')] for line in file.readlines()]
            return encodings

        encodings = load_encodings(user_encodings)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int (cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0

        frames_to_video = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            #Vamos realizar validações frame a frame, gerando um novo vídeo para extração das emoções, com os seguintes requisitos:

                # 1 - Se não houver nenhum rosto no frame, ele adiciona um frame branco ao novo vídeo
            
                # 2 - Se houver rostos, ele irá percorrer todos, verificando se algum deles é o da pessoa procurada
            
                # 3 - Caso algum for, ele adiciona o frame com o rosto ao novo vídeo, e vai para o próximo frame
            
                # 4 - Caso nenhum rosto encontrado for da pessoa procurada, ele adiciona um frame branco ao novo vídeo
            
            img2 = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            faces = face_recognition.face_locations(img2)
            
            if not faces:
                white = cv2.resize(white_image, (144, 144))
                frames_to_video.append(white)

            else:
                user_found = False

                for face_locations in faces:
                    encodeFace = face_recognition.face_encodings(img2, [face_locations])[0]
                    match = face_recognition.compare_faces(encodings, encodeFace, 0.5)

                    if match[0]:
                        y1, x2, y2, x1 = face_locations
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        face_roi = frame[y1:y2, x1:x2]
                        face_roi = cv2.resize(face_roi, (144, 144))
                        frames_to_video.append(face_roi)
                        user_found = True
                        break  

            if not user_found:
                white = cv2.resize(white_image, (144, 144))
                frames_to_video.append(white)                                    
            
            processed_frames += 1

            print('Frames processados:', processed_frames, 'de', total_frames)

            if processed_frames == total_frames:
                break     

        print('Configurando saída de vídeo...')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('video_cut.mp4', fourcc, fps, (144, 144))
        
        for frame in frames_to_video:
            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    except Exception as e:
        print('Erro durante o processamento do novo vídeo:', e)
    
    try:
        #Após gerar o novo vídeo, ele inicia a detecção e extração das emoções com Py-Feat, salvando em um arquivo CSV
        print('Iniciando detecção de emoções...')
        detector = Detector(identity_model='facenet', device='cuda')

        video_cut_path = 'video_cut.mp4'

        video_prediction = detector.detect_video(video_cut_path)

        output_video = video_prediction.dropna()

        output_video = video_prediction.loc[:, ['approx_time', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']]

        df = pd.DataFrame(output_video)

        grouped = df.groupby('approx_time').mean()

        print('Salvando emoções no CSV...')
        grouped.to_csv(f'/home/daniel/Documentos/IA/teste/results/'+user_name + '_emotions.csv')

        os.remove(video_cut_path)

        stop = time.time()

        print('Tempo de processamento: ' + str(stop - start) + ' segundos')

    except Exception as e:
        print('Erro durante a tentativa de detecção:', e)

if __name__ == '__main__':
    fex('/home/daniel/Documentos/IA/teste/video.mp4', '/home/daniel/Documentos/IA/teste/encodings/rubens_encodings.txt', 'Rubens')
