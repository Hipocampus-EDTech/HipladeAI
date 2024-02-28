import cv2
import face_recognition

def extract(image_path):

    img = cv2.imread(image_path)
    
    encode = face_recognition.face_encodings(img)[0]
    
    return encode

if __name__ == '__main__':
    image_path = '/home/daniel/Documentos/IA/Fex/imgs/Rubens.jpeg'
    user_name = 'rubens'
    encoding = extract(image_path)

    with open(f'encodings/{user_name}_encodings.txt', 'w') as f:
        f.write(','.join(str(x) for x in encoding) + '\n')
