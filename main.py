import cv2
import mediapipe as mp
from analyze_image import check_image_for_jaw_openness, image_from_path
from draw import draw_landmarks_on_image, draw_image, DrawingStyle
import numpy as np





def process_frame(image: mp.Image,jaw_openness_threshold, drawing_style ) -> np.ndarray:
    """
    Process a single frame by checking jaw openness, drawing landmarks, and displaying the processed image.

    Args:
        image (mp.Image): Input image.

    Returns:
        mp.Image: Processed image.
    """

    detection_result = check_image_for_jaw_openness(image)
    image = image.numpy_view()
    image = draw_landmarks_on_image(image, detection_result, drawing_style=drawing_style)
    
    # Get the openness of the jaw
    openness_of_jaw = detection_result.face_blendshapes[0][25].score
    processed_image = draw_image(openness_of_jaw, image=image, jaw_openness_threshold=jaw_openness_threshold)
    return processed_image

def run_video(video_path: str, jaw_openness_threshold, drawing_style) -> None:
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
            processed_image = process_frame(frame, jaw_openness_threshold, drawing_style)
            # Display the processed image
            cv2.imshow("Video Window", processed_image)
            # Close the video window on the press of 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cam.release()
    cv2.destroyAllWindows()

def run_image(image: mp.Image, jaw_openness_threshold, drawing_style) -> None:
    """
    Run image processing using the analyze_image module and display the processed image.

    Args:
        image (mp.Image): Input image.
    """
    
    processed_image = process_frame(image, jaw_openness_threshold, drawing_style)
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image Window", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def take_input() -> tuple[str, str, float, DrawingStyle]:
    """
    Take input from the user.

    Returns:
        Tuple[str, str, float, DrawingStyle]: A tuple containing the type of input, the path to the input, the threshold for jaw openness, and the drawing style.
    """

    print("press q to exit the window created by the program")
    type = input("Enter the type of input (image/video): ")

    path = input("Enter the path to the input file: ")
    custom_jaw_openness_threshold = input("Do you want to enter a custom threshold for jaw openness? (y/n): ").lower()
    if custom_jaw_openness_threshold == "y":
        jaw_openness_threshold = float(input("Enter the threshold for jaw openness: "))

    elif custom_jaw_openness_threshold == "n":
        jaw_openness_threshold = 0.007
    
    else:
        print("Invalid input")
        exit()

    custom_drawing_style = input("Do you want to enter a custom drawing style? (y/n): ").lower()

    if custom_drawing_style == "y":
        drawing_style = input("Enter the drawing style (TESSELATION/CONTOURS/IRIS): ").upper()
    elif custom_drawing_style == "n":  
        drawing_style = None
    else:
        print("Invalid input")
        exit()

    if drawing_style is not None:
        drawing_style = DrawingStyle[drawing_style]
    return type, path, jaw_openness_threshold, drawing_style

def main(type, path, jaw_openness_threshold = 0.007, drawing_style = None) -> None:
    """
    Main function to demonstrate image processing on both image and video.
    """
    if type == "video":
        video_path = path
        run_video(video_path, jaw_openness_threshold, drawing_style)
    
    elif type =="image":
        image_path = path
        image = image_from_path(image_path)
        run_image(image, jaw_openness_threshold, drawing_style)
    else:
        print("Invalid type")

if __name__ == "__main__":
    type, path, jaw_openness_threshold, drawing_style = take_input()
    main(type, path, jaw_openness_threshold, drawing_style)