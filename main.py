import analyze_image
import cv2
import mediapipe as mp


def run_video(video_path):
  cam = cv2.VideoCapture(video_path)
  while True:
      ret, frame = cam.read()
      if ret:
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        openess_of_jaw,annotated_image = analyze_image.check_image_openness(frame,TESSELATION = False, CONTOURS = False, IRIS = False )
        image = analyze_image.draw_image(openess_of_jaw, 0.01,annotated_image)

        #close video on the press of q
        cv2.imshow("windwoname",image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

      else:
        break
  cam.release()


def run_image(image_path):
  image = analyze_image.image_from_path(image_path)
  openess_of_jaw,annotated_image = analyze_image.check_image_openness(image, TESSELATION = False, CONTOURS = False, IRIS = False)
  image = analyze_image.draw_image(openess_of_jaw, 0.01,annotated_image)
  cv2.imshow("windwoname",cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def main():

  run_image("/home/sonnet/oralens/cat2/test_resources/man.jpg")


if __name__ == "__main__":
    main()