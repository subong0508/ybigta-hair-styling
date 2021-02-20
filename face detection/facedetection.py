# ------------------------------------------------------------------------------
# Created by Seungone Kim(louisdebroglie@yonsei.com)
# YBIGTA 2021 spring project : ybigta-hair-styling
# ------------------------------------------------------------------------------

import dlib
import face_recognition as fr
import cvlib
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image 
from skimage import io
from globals import ALL,JAWLINE,RIGHT_EYEBROW,LEFT_EYEBROW,FACESHAPE

def dlib_process():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    return (detector,predictor)

def CropInSquare(image_path):
    '''
    This function crops the face in a square estimating the location of the face.
    This function assumes that there is only onee face in the picture.
    
    @param image_path (str): the absolute path of where your image is stored.

    @returns image (image): returns the cropped part of the FaceShape as a RGB image.
    '''
    # preparing detector and predictor
    detector,_ = dlib_process()

    # loading the image
    image = dlib.load_rgb_image(image_path)

    # using the shape_detector
    faces = detector(image,1)
    face_frames = [(x.left(), x.top(), x.right(), x.bottom()) for x in faces]

    # cropping the image based on the left,right,bottom,top coordinate info
    for idx,face_rect in enumerate(face_frames):
        image = Image.fromarray(image).crop(face_rect)

    return image

def CropInFaceShape(image_path):
    '''
    This function crops the face in the FaceShape using shape_predictor_68_face_landmarks.dat
    Specifically, we use the JAWLINE, LEFT_EYEBROW, RIGHT_EYEBROW to crop into the FaceShape.
    This function assumes that there is only one face in the picture.
    However, if you change the for loop using faces, you might be able to get multiple cropped faces.

    @param image_path (str): the absolute path of where your image is stored.

    @returns image (image): returns the cropped part of the FaceShape as a RGB image.
    @returns out (image): mask to show which part of the face was cropped like image segmentation
    @returns mask (image): showing the 68 dots in the image and how the face was cropped.
    '''
    # preparing detector and predictor
    detector,predictor = dlib_process()
    
    # loading the image, then resizing and changing the color.
    image = dlib.load_rgb_image(image_path)
    image = cv2.resize(image, dsize=(640,480), interpolation=cv2.INTER_AREA)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # using the shape_detector
    faces = detector(image_gray,1)

    # accessing to each faces and getting the coordinates needed to crop into FaceShape.
    landmarklist_local =[]
    for i,face in enumerate(faces):
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 3)
        landmarks = predictor(image, face)
        landmark_list=[]
        for p in landmarks.parts():
            landmark_list.append([p.x, p.y])
            cv2.circle(image, (p.x, p.y),2,(0,255,0),-1)
        #landmark_list contains coordinates of each face, and landmarklist_local contains every landmarklist_local
        landmarklist_local.append(landmark_list)

    # Changing the image color back to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # For those looking at my github,
    # You need to change this part to use this function to crop multiple faces.
    # use a for loop to access to every face in landmarklist_local
    # Good Luck!
    landmark_tuple = landmarklist_local[0]

    # Storing the information of the coordinates of the FaceShape in order to crop.
    routes=[]
    # coorinates in JAWLINE
    for i in range(15,-1,-1):
        from_coordinate = landmark_tuple[i+1]
        to_coordinate = landmark_tuple[i]
        routes.append(from_coordinate)
    from_coordinate = landmark_tuple[0]
    to_coordinate = landmark_tuple[17]
    routes.append(from_coordinate)
    # coordinates in RIGHT_EYEBROW
    for i in range(17,20):
        from_coordinate = landmark_tuple[i]
        to_coordinate = landmark_tuple[i+1]
        routes.append(from_coordinate)
    from_coordinate = landmark_tuple[19]
    to_coordinate = landmark_tuple[24]
    routes.append(from_coordinate)
    # coordinates in LEFT_EYEBROW
    for i in range(24,26):
        from_coorindate = landmark_tuple[i]
        to_coordinate = landmark_tuple[i+1]
        routes.append(from_coordinate)
    from_coordinate = landmark_tuple[26]
    to_coordinate = landmark_tuple[16]
    routes.append(from_coordinate)
    routes.append(to_coordinate)

    # adding line to crop the image
    for i in range(0, len(routes)-1):
        from_coordinate = routes[i]
        to_coordinate = routes[i+1]
        image = cv2.line(image,tuple(from_coordinate),tuple(to_coordinate),(255,255,0),1)

    # Changing the image color back to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # making mask to show which part of the face was cropped like image segmentation
    mask = np.zeros((image.shape[0], image.shape[1]))
    mask = cv2.fillConvexPoly(mask, np.array(routes),1)
    mask = mask.astype(np.bool)

    # showing the 68 dots in the image and how the face was cropped.
    out = np.zeros_like(image)
    out[mask] = image[mask]

    return image, out, mask