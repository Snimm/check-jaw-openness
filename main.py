import cv2
import mediapipe as mp
from analyze_image import check_image_openness, draw_image, image_from_path

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
            openness_of_jaw, annotated_image = check_image_openness(frame, TESSELATION=False, CONTOURS=False, IRIS=False)
            processed_image = draw_image(openness_of_jaw, image=annotated_image, jaw_openness_threshold=0.01)

            # Display the processed image
            cv2.imshow("WindowName", processed_image)

            # Close the video window on the press of 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cam.release()

def run_image(image_path: str) -> None:
    """
    Run image processing using the analyze_image module.

    Args:
        image_path (str): The path to the input image file.
    """
    image = image_from_path(image_path)
    openness_of_jaw, annotated_image = check_image_openness(image, TESSELATION=False, CONTOURS=False, IRIS=False)
    processed_image = draw_image(openness_of_jaw, image=annotated_image, jaw_openness_threshold=0.01)

    # Display the processed image
    cv2.imshow("WindowName", cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main() -> None:
    """
    Main function to demonstrate image processing.
    """
    run_image("/home/sonnet/oralens/cat2/test_resources/man.jpg")
    run_video("/home/sonnet/oralens/cat2/test_resources/Sequence_13.mp4")

if __name__ == "__main__":
    main()
