import cv2
import media_pipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

model_path = 'face_landmarker.task'

def image_from_path(image_path):
    image = mp.Image.create_from_file(image_path)
    return image



def check_image_openness(image,TESSELATION = False, CONTOURS = False, IRIS = False):


    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)


    detection_result = detector.detect(image)

    annotated_image = media_pipe.draw_landmarks_on_image(image.numpy_view(), detection_result,TESSELATION, CONTOURS, IRIS)

    openess_of_jaw = detection_result.face_blendshapes[0][25].score
    return openess_of_jaw,annotated_image
    #print(detection_result.face_blendshapes[0])
    # media_pipe.plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0])
    #print(detection_result.facial_transformation_matrixes)

def draw_image(openess_of_jaw, jaw_openness_threshold, annotated_image):
    if openess_of_jaw > jaw_openness_threshold:
        jaw_class = "open"
    else:
        jaw_class = "close"
        
    text_to_put = f"{str(openess_of_jaw)[:5]}  {jaw_class}"
    text_location = (0, annotated_image.shape[0]//40+10)



    cv2.putText(annotated_image,text_to_put,text_location,cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),4,cv2.LINE_AA)
    cv2.putText(annotated_image,text_to_put,text_location,cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),1,cv2.LINE_AA)
    return annotated_image

