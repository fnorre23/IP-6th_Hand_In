from deepface import DeepFace as dp
import cv2 
import numpy as np
import socket
import os

#Communication with Unity ####################################################################
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
serverAddressPort = ("127.0.0.1", 5052)

def displayText(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (10, 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return image

def displayTextBelow(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (10, 65), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return image

def get_emotion(frame):

    # Save the image
    cv2.imwrite('emotion_image.png', frame)
    im_path= 'emotion_image.png'

    # facial analysis
    objs = dp.analyze(
    img_path = im_path, 
    actions = ['emotion'],
    )

    # Extracting emotion 
    for obj in objs:
        dominant_emotion = obj['dominant_emotion']
    
    # Send emotion to Unity
    sock.sendto(str.encode(dominant_emotion), serverAddressPort)

    return dominant_emotion

# Using webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image")
        break

    raw_frame = cv2.flip(frame, 1)  # Flip the frame horizontally

    # Close by pressing q
    key = cv2.waitKey(3) & 0xFF
    if key == ord('q'):
        break

    if key == ord('w'):
        emotion = get_emotion(frame)

    # Check if variable exists, and then display correct emotion
    if 'emotion' in locals():
        frame = displayText(frame, emotion)
    else: 
        frame = displayText(frame, 'No Emotion')
    
    # Displaying instructions
    frame = displayTextBelow(frame, 'Press w to calc emotion')

    cv2.imshow('Live feed', frame)    


cap.release()
cv2.destroyAllWindows()
sock.close()

# Delete the original captured image
try:
    os.remove('emotion_image.png')
    print("Original image deleted.")
except FileNotFoundError:
    print("Error: Original image file not found.")
except Exception as e:
    print(f"Error deleting the image: {e}")
