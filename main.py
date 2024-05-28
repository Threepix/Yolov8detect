import argparse
from yolo_detector import YOLODetector
from utils import process_video

def main(input_path, output_path):
    detector = YOLODetector()
    process_video(input_path, output_path, detector)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 People Detection in Video")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, required=True, help="Path to output video file")
    
    args = parser.parse_args()
    main(args.input, args.output)
