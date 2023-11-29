# Open Jaw detection with OpenCV and MediaPipe

This Python program utilizes OpenCV and MediaPipe to perform face analysis on images and videos. The main functionalities include detecting landmarks on the face, checking jaw openness, and providing visualizations based on user-defined parameters.

## Installation

Before running the program, make sure to install python


## Operational Procedure
```bash
# Clone the repository
git clone https://github.com/Snimm/check-jaw-openness

# Navigate to the project directory
cd check-jaw-openness

# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # For Linux/macOS
# or
.\.venv\Scripts\activate  # For Windows

# Install the required dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

3. Follow the on-screen prompts to choose the type of input (image or video), provide the file path, and customize the analysis parameters.

4. The program will display the processed output, showcasing facial landmarks and indicating jaw openness.

## Customization

- **Threshold for Jaw Openness:** Users have the option to either input a custom threshold for jaw openness or utilize the default value, particularly beneficial when the user prefers to exercise an abundance of caution.


- **Drawing Style:** Choose a drawing style for facial landmarks visualization, including TESSELATION, CONTOURS, IRIS, or None for no additional visualization.

## Examples

### Video Processing

```bash
python main.py
Enter the type of input (image/video): video
Enter the path to the input file: path/to/your/video.mp4
Do you want to enter a custom threshold for jaw openness? (y/n): y
Enter the threshold for jaw openness: 0.01
Do you want to enter a custom drawing style? (y/n): y
Enter the drawing style (TESSELATION/CONTOURS/IRIS): CONTOURS
```

### Image Processing

```bash
python main.py
Enter the type of input (image/video): image
Enter the path to the input file: path/to/your/image.jpg
Do you want to enter a custom threshold for jaw openness? (y/n): n
Do you want to enter a custom drawing style? (y/n): y
Enter the drawing style (TESSELATION/CONTOURS/IRIS): TESSELATION
```
*Example output*

![Image Window_screenshot_30 11 2023](https://github.com/Snimm/check-jaw-openness/assets/53926889/a3a702aa-be17-4057-9c39-cc21b75858a3)


##Insight Guide

**Understanding the Assignment:**

The primary objective of this project is to develop an algorithm capable of analyzing images or videos of faces and classifying each frame based on whether the subject's jaw is open or closed.

**Implementation Strategy:**

1. **Fragment Video into Frames:**
   - Extract frames from the input video. Each frame is treated as an independent unit for jaw openness classification.

2. **Extract Face Landmarks:**
   - Utilize Media Pipe's face_landmarker model to identify and extract all facial landmarks present in each frame.

3. **Get Jaw Openness from Landmarks:**
   - Directly extract the openness of the jaw from face blendshapes provided by MediaPipe based on the facial landmarks.

4. **Classification Based on Jaw Openness:**
   - Implement a classification step where frames are categorized as either having an open or closed jaw, determined by a predefined threshold of jaw openness.

5. **Display Openness and Classification:**
   - In the top-left corner of each processed image, showcase both the calculated openness of the jaw and the corresponding classification (open/closed).

6. **Display Landmarks (Optional):**
   - Offer the option to display facial landmarks in different styles such as TESSELATION, CONTOURS, or IRIS for visual reference.

7. **Documentation:**
   - I have ensure that the code is well-documented throughout the development process, with clear explanations and comments to facilitate understanding.

