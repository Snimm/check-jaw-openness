import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import media_pipe

# Path to the face landmarker model
MODEL_PATH = 'face_landmarker.task'

def image_from_path(image_path: str) -> mp.Image:
    """
    Create an image from the given file path.

    Args:
        image_path (str): The path to the input image file.

    Returns:
        mp.Image: The created image.
    """
    image = mp.Image.create_from_file(image_path)
    return image

def check_image_openness(image: mp.Image, TESSELATION: bool = False, CONTOURS: bool = False, IRIS: bool = False) -> tuple[float, mp.Image]:
    """
    Check the openness of the jaw in the given image.

    Args:
        image (mp.Image): The input image.
        TESSELATION (bool): Flag to include tessellation in the annotated image.
        CONTOURS (bool): Flag to include contours in the annotated image.
        IRIS (bool): Flag to include iris landmarks in the annotated image.

    Returns:
        Tuple[float, np.ndarray]: A tuple containing the openness of the jaw and the annotated image.
    """
    # Create base options for the face landmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    # Specify options for face landmarker
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    # Create the face landmarker
    detector = vision.FaceLandmarker.create_from_options(options)

    # Perform face detection on the image
    detection_result = detector.detect(image)

    # Draw landmarks on the image
    annotated_image = media_pipe.draw_landmarks_on_image(image.numpy_view(), detection_result, drawing_style = None)

    # Get the openness of the jaw
    openness_of_jaw = detection_result.face_blendshapes[0][25].score

    return openness_of_jaw, annotated_image

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
    cv2.putText(image, text_to_put, text_location, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(image, text_to_put, text_location, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

    return image
