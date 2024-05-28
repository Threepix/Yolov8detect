# YOLOv8 People Detection in Video

This project performs people detection in a given video file using the YOLOv8 model. The detected people are highlighted in the video with bounding boxes that include the class name and confidence score.

## Setup

Download crowd.mp4 from https://cloud.mail.ru/public/431z/MhGUAfGb2

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Run

Убедитесь, что у вас есть видеофайл crowd.mp4 в директории проекта. Запустите проект с помощью следующей команды:

```bash
python main.py --input crowd.mp4 --output crowd_detected.mp4
```
