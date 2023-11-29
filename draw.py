import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class DrawingStyle(Enum):
    TESSELATION = 'TESSELATION'
    CONTOURS = 'CONTOURS'
    IRIS = 'IRIS'

def draw_landmarks_on_image(rgb_image, detection_result, drawing_style=None):
    """
    Draw landmarks on the input image.

    Args:
        rgb_image: The RGB image.
        detection_result: The result of face landmark detection.
        drawing_style (DrawingStyle): The style for drawing landmarks.

    Returns:
        np.ndarray: Annotated image with landmarks.
    """
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        if drawing_style == DrawingStyle.TESSELATION:
            mp.solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            )
        elif drawing_style == DrawingStyle.CONTOURS:
            mp.solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
            )
        elif drawing_style == DrawingStyle.IRIS:
            mp.solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
            )

    return annotated_image


def draw_image(openess_of_jaw: float, image: mp.Image, jaw_openness_threshold: float) -> mp.Image:
    """
    Draw the openness of the jaw on the image.

    Args:
        openess_of_jaw (float): The openness of the jaw.
        image (np.ndarray): The input image.
        jaw_openness_threshold (float): The threshold for considering the jaw open.

    Returns:
        np.ndarray: The image with openness information drawn.
    """
    # Determine jaw class based on the threshold
    jaw_class = "open" if openess_of_jaw > jaw_openness_threshold else "close"
    
    # Create text to put on the image
    text_to_put = f"{str(openess_of_jaw)[:5]}  {jaw_class}"
    text_location = (0, image.shape[0] // 40 + 10)

    # Put text on the image
    put_text(image, text_to_put, text_location, size=.5)

    return image

def put_text(image, text_to_put, text_location, size):
    cv2.putText(image, text_to_put, text_location, cv2.FONT_HERSHEY_COMPLEX, size, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(image, text_to_put, text_location, cv2.FONT_HERSHEY_COMPLEX, size, (0, 0, 0), 1, cv2.LINE_AA)