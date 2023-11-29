import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import draw

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

def check_image_for_jaw_openness(image: mp.Image) -> tuple[float, mp.Image]:
    """
    Check the openness of the jaw in the given image.

    Args:
        image (mp.Image): The input image.

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

    # Draw landmarks on the image (add your drawing logic here if needed)

    return detection_result
