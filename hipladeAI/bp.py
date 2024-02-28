import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import ffmpegio

def apply2Video(callFunction, videoPath="2023-09-12 18-34-56.mp4", cvt=None,**kwargs):
    '''
    Function that applies to every frame  of the video the 'callFunction'.
    ---------------------------------------------------------------------
    callFunction = function that 
    videoPath = path to the video
    cvt = colorsecheme  -Optional  (cv2.COLOR_RGB2GRAY,cv2.COLOR_BGR2RGB, cv2.COLOR_BGR2HSV, ....)
    
    ---------------------------------------------------------------------
    Returns a list with the callback output
    '''
    cap = cv2.VideoCapture(videoPath)
    output = []
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break
        if not(cvt is None):
            image = cv2.cvtColor(image,cvt)
        output.append(callFunction(image,**kwargs))
    cap.release()
    return output

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks])
        solutions.drawing_utils.draw_landmarks(
              annotated_image,
              pose_landmarks_proto,
              solutions.pose.POSE_CONNECTIONS,
              solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


# contruindo detector (global)
def getDetector():
    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)
    return detector

def getBonesImages(frame,detector):
    # criando imagem do tipo MediaPipe e aplicando o detector
    mpFrame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    results = detector.detect(mpFrame)
    return draw_landmarks_on_image(frame,results)
    
def euclid(arr):
    return np.sqrt(np.sum(arr**2))

def getCoord(landmark):
    return np.array([landmark.x,landmark.y,landmark.z])
    
def b2p(detection_result):
    metric = []
    points = detection_result.pose_landmarks[0]
    set1 = [points[16].visibility,points[14].visibility,points[12].visibility,points[24].visibility]
    set2 = [points[15].visibility,points[13].visibility,points[11].visibility,points[23].visibility]
    if np.average(set1)>np.average(set2):
        hand,elbow,shoulder,hips = getCoord(points[16]),getCoord(points[14]),getCoord(points[12]),getCoord(points[24])
    else:
        hand,elbow,shoulder,hips = getCoord(points[15]),getCoord(points[13]),getCoord(points[11]),getCoord(points[23])
    v1 = hips-shoulder
    v2 = elbow-shoulder
    metric.append(np.dot(v1,v2)/(euclid(v1)*euclid(v2)))
    
    v1 = hand-elbow
    v2 = shoulder-elbow
    metric.append(np.dot(v1,v2)/(euclid(v1)*euclid(v2)))
    return np.average(metric)

def b1p(detection_result):
    metric = []
    points = detection_result.pose_landmarks[0]
    set1 = [points[12].visibility,points[24].visibility,points[26].visibility]
    set2 = [points[11].visibility,points[23].visibility,points[25].visibility]
    if np.average(set1)>np.average(set2):
        shoulder,hips,knee = getCoord(points[12]),getCoord(points[24]),getCoord(points[26])
    else:
        shoulder,hips,knee = getCoord(points[11]),getCoord(points[23]),getCoord(points[25])
    v1 = shoulder-hips
    v2 = knee-hips
    metric.append(np.dot(v1,v2)/(euclid(v1)*euclid(v2)))
    
    return metric

