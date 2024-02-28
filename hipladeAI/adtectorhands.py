import mediapipe as mp
import pandas as pd
import cv2
from shapely.geometry import Polygon, Point
from math import sqrt

# Declaração de variaveis
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

a, b, c = None, None, None
i, j, k = None, None, None
centroid1, centroid2 = None, None
area1, area2 = None, None

# Criar uma lista vazia para armazenar os dados
dados = []

def adetectorOnline(inputCamera=0):
	

	# Função de encontrar o primeiro triangulo baseado nos três pontos da mão direita
	def triangleOne():
		global a, b, c, i, j, k, area1, centroid1, area2, centroid2

		a = (int(hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * width), int(hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * height))				  
		b = (int(hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width), int(hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height))
		c = (int(hand_landmark.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width), int(hand_landmark.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height))
		area1 = Polygon([a, b, c]).area
		centroid1 = tuple(Polygon([a, b, c]).centroid.coords)[0]
		cv2.circle(video, (int(centroid1[0]), int(centroid1[1])), 3, (255, 255, 255), -1)	  
		cv2.line(video, a, b, (0, 0, 255), 2)
		cv2.line(video, b, c, (0, 0, 255), 2)
		cv2.line(video, a, c, (0, 0, 255), 2)
		cv2.putText(video, 'RIGHT', (570, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		return centroid1

	# Função de encontrar o primeiro triangulo baseado nos três pontos da mão esquerda
	def triangleTwo():

		global i, j, k, area2, centroid2
		i = (int(hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * width), int(hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * height))				  
		j = (int(hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width), int(hand_landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height))
		k = (int(hand_landmark.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width), int(hand_landmark.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height))
		area2 = Polygon([i, j, k]).area
		centroid2 = tuple(Polygon([k, i, j]).centroid.coords)[0]
		cv2.circle(video, (int(centroid2[0]), int(centroid2[1])), 3, (255, 255, 255), -1)
		cv2.line(video, i, j, (0, 0, 0), 2)
		cv2.line(video, i, k, (0, 0, 0), 2)
		cv2.line(video, j, k, (0, 0, 0), 2)			
		cv2.putText(video, 'LEFT', (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

		return centroid2

	# Função para calcular o score
	def score():
		global a, b, c, i, j, k, area1, centroid1, area2, centroid2
		m = 0
		if area1 is not None and area2 is not None:
			m = abs((area1 - area2) - Point(centroid1).distance(Point(centroid2))) / (Polygon([(0, 0), (0, video.shape[0]), (video.shape[1], 0)]).area + sqrt((video.shape[0] ** 2) + (video.shape[1] ** 2)))
			cv2.line(video, (int(centroid1[0]), int(centroid1[1])), (int(centroid2[0]), int(centroid2[1])), (255, 255, 255), 1)
			cv2.circle(video, (int((centroid1[0] + centroid2[0]) / 2), int((centroid1[1] + centroid2[1]) / 2)), 2, (255, 255, 255), -1)
		
		return m

	# Captura de vídeo/imagem ou tempo real
	capture = cv2.VideoCapture(inputCamera) #'/home/daniel/ADTector2/input-correto/001.mp4'

	# Inicia detecção de mãos dentro de um loop
	with mp_hands.Hands(static_image_mode= True, max_num_hands=2, min_detection_confidence= 0.5) as hands:
		while True:  
			ret, video = capture.read() 

			# Verificar se o frame foi lido corretamente
			if not ret:
				break 
			
			# Valida se não ouve erro ao tentar abrir o vídeo
			if not capture.isOpened():
				print("Erro ao abrir o vídeo.")
				exit()

			# Video		
			image_rgb = cv2.cvtColor(video, cv2.COLOR_BGR2RGB)
			height, width, _ = video.shape

			# Image
			# image = cv2.imread("/home/daniel/Documentos/Workshop/IA/hands/003.png")
			# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			# height, width, _ = image.shape

			# Variavel com o resultado do processamento das mãos na imagem passada
			results = hands.process(image_rgb)

			# Valida se existe alguma mão na imagem
			if results.multi_hand_landmarks is not None:
				for hand_landmark, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
					#mp_drawing.draw_landmarks(image, hand_landmark, mp_hands.HAND_CONNECTIONS)  
					
					# Chama função da avaliação do triangulo 1
					if handedness.classification[0].label == 'Left':
						triangleOne()					 
					
					# Chama função da avaliação do triangulo 2
					if handedness.classification[0].label == 'Right':
						triangleTwo() 

			cv2.putText(video, ("Score: " + str(score())), (20, 20), 1, 1, (255, 0, 0), 2)

			cv2.putText(video, 'A: {}'.format(a), (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(video, 'B: {}'.format(b), (400, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(video, 'C: {}'.format(c), (400, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

			cv2.putText(video, 'I: {}'.format(i), (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
			cv2.putText(video, 'J: {}'.format(j), (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
			cv2.putText(video, 'K: {}'.format(k), (80, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

			cv2.imshow('ADTector', video)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
if __name__=='__main__':
	adetectorOnline()
