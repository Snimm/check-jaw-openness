# Open Jaw detection with OpenCV and MediaPipe

This Python program utilizes OpenCV and MediaPipe to perform face analysis on images and videos. The main functionalities include detecting landmarks on the face, checking jaw openness, and providing visualizations based on user-defined parameters.

## Installation

Before running the program, make sure to install python


## Usage
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

- **Threshold for Jaw Openness:** Users can input a custom threshold for jaw openness or use the default value.

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
![Image Window_screenshot_30 11 2023](https://github.com/Snimm/check-jaw-openness/assets/53926889/a3a702aa-be17-4057-9c39-cc21b75858a3)

