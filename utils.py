import cv2
import numpy as np

def process_video(input_path, output_path, detector):
    """
    Processes the input video and saves the output video with detected people.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the output video file.
        detector (YOLODetector): Instance of YOLODetector for detection.
    
    Returns:
        bool: True if processing was successful, False otherwise.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_path}")
        return False

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        for detection in detections:
            xmin, ymin, xmax, ymax, confidence, cls = detection
            if int(cls) == 0:  # Class 0 is 'person' in COCO dataset
                xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
                label = f"person {confidence:.2f}"
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return True
