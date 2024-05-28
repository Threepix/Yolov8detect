import cv2

def process_video(input_path, output_path, detector):
    """
    Processes the input video and saves the output video with detected people.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the output video file.
        detector (YOLODetector): Instance of YOLODetector for detection.
    """
    cap = cv2.VideoCapture(input_path)
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
        for _, row in detections.iterrows():
            if row['name'] == 'person':
                xmin, ymin, xmax, ymax, confidence = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence']
                label = f"{row['name']} {confidence:.2f}"
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
