import cv2
import mediapipe as mp
from analyze_image import check_image_for_jaw_openness, image_from_path
from draw import draw_landmarks_on_image, draw_image, DrawingStyle
import numpy as np

def process_frame(image: mp.Image) -> np.ndarray:
    """
    Process a single frame by checking jaw openness, drawing landmarks, and displaying the processed image.

    Args:
        image (mp.Image): Input image.

    Returns:
        mp.Image: Processed image.
    """
    detection_result = check_image_for_jaw_openness(image)
    # image = draw_landmarks_on_image(image.numpy_view(), detection_result, drawing_style=None)
    
    image = np.copy(image.numpy_view())
    # Get the openness of the jaw
    openness_of_jaw = detection_result.face_blendshapes[0][25].score
    processed_image = draw_image(openness_of_jaw, image=image, jaw_openness_threshold=0.01)
    return processed_image

def run_video(video_path: str) -> None:
    """
    Run video processing using the analyze_image module.

    Args:
        video_path (str): The path to the input video file.
    """
    cam = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cam.read()
        if ret:
            frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            processed_image = process_frame(frame)
            # Display the processed image
            cv2.imshow("Video Window", processed_image)
            # Close the video window on the press of 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cam.release()
    cv2.destroyAllWindows()

def run_image(image: mp.Image) -> None:
    """
    Run image processing using the analyze_image module and display the processed image.

    Args:
        image (mp.Image): Input image.
    """
    processed_image = process_frame(image)
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image Window", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main() -> None:
    """
    Main function to demonstrate image processing on both image and video.
    """
    image_path = "/home/sonnet/oralens/cat2/test_resources/man.jpg"
    video_path = "/home/sonnet/oralens/cat2/test_resources/Sequence_13.mp4"
    
    # Process and display a single image
    image = image_from_path(image_path)
    run_image(image)

    # Process and display video frames
    run_video(video_path)

if __name__ == "__main__":
    main()
