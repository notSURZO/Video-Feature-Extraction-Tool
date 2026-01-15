# Video-Feature-Extraction-Tool
## A python script which reads a local video file and outputs the important features.

This project designs a comprehensive Python script that is able to process video files in a local directory and extract important visual and temporal metadata by combining computer vision techniques and deep learning models. It extracts- 

*  Hard Cut Detection: Hard cut detection was done by calculating the color histogram for every frame and comparing it to the previous frame using correlation. A cut is detected if the histogram similarity is less than a user set threshold. It ignores minor fluctuations within a short frame window which can also be configured by the user.
  
* Motion Analysis: In motion analysis it uses Farneback Dense Optical Flow Algorithm. The mean magnitude of the movement vectors between consecutive frames were considered as the motion value for that pair of frames.
* Text Detection (OCR): Deep Learning-based Optical Character Recognition model EasyOCR was used to determine a ratio of the presence of texts in the video as well as extract the keywords from those texts.

* Object vs. Person Dominance: A pre-trained deep learning based model, YOLOv8m was used with ByteTracking to detect and track objects as well as humans and calculate the ratio of people and objects.


---



![Image](https://github.com/user-attachments/assets/93a952fc-9c05-46a6-96f1-4b75dae18909)


### How to run the code

---

1. Install the requirements by running this cell block

```
!pip install ultralytics supervision easyocr opencv-python-headless
```


2. Import necessary libraries and models

```

import numpy as np
from ultralytics import YOLO
import time
import cv2
import easyocr
import json

```


3. Run the main function's definition block. 

    Set ```gpu = false``` in the ``` ocr = easyocr.Reader(...)``` if you don't have a dedicated gpu (gpu is recommended). 


```
def main(src_path, output_json, hist_diff_threshold = 0.8, ocr_confidence_threshold = 0.6, word_count = 2, cut_frames_threshold = 10, ocr_div = 2,):
    

    model = YOLO("yolov8m.pt")
    ocr = easyocr.Reader(['en'], gpu=True)

    people_list = []
    ...
    ...
    ...
```
4. Finally, Set all your necessary paths, thresholds, values and run this block.

    ```src_path``` - Path to your local video file
    
    ```output_json``` - Path to the JSON file for output

    ```hist_diff_threshold``` - How much sensitivity you want for histogram difference

    ```ocr_confidence_threshold``` - Confidence above which texts get detected; 

    ```word_count``` - How many of OCR the detected words you want to show

    ```cut_frames_threshold``` -> For how many frames a cut is to be considered

```
if __name__ == "__main__":

    src_path = '/content/input_video.mp4'

    output_json = '/content/features.json'


    hist_diff_threshold = 0.8

    ocr_confidence_threshold = 0.6

    word_count = 2

    cut_frames_threshold = 10

    ocr_div = 2


    main(src_path, output_json, hist_diff_threshold, ocr_confidence_threshold, word_count, cut_frames_threshold, ocr_div)

```

---

The output will be printed in the terminal in a JSON format as well as be saved to the dedicated JSON output file.

```
{
    "People versus objects ratio": "1 : 4",
    "No. of hard cuts": 2,
    "Mean motion": " 0.5498",
    "Text Presence Ratio": " 0.00% )"
}
```

### Details about the OCR model [Read here](https://github.com/JaidedAI/EasyOCR)

### Details about the YOLO model [Read here](https://github.com/ultralytics/ultralytics)


---
