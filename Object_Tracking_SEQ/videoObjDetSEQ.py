import cv2
import time

import line_profiler
from ultralytics import YOLO

yolo = YOLO('yolov8n.pt')

video_path = ""

videoCap = cv2.VideoCapture(video_path if video_path else 0)

if not videoCap.isOpened():
    raise ValueError("ERROR: Video is not readable. Make sure path is correct and try again.")

desired_classes = ["car"]
all_classes = True


def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] *
             (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

processed_frames = 0
t_proc_start = time.time()
while True:

    fps = videoCap.get(cv2.CAP_PROP_FPS)
    sleep_time = 1 / fps if fps > 0 else 0.03
    ret, frame = videoCap.read()
    time.sleep(sleep_time)

    if not ret:
        break

    results = yolo.predict(frame, stream=True)
    detections = []
    for result in results:
        classes_names = result.names
        for box in result.boxes:
            if box.conf[0] > 0.4:
                cls = int(box.cls[0])
                class_name = classes_names[cls]
                if not all_classes and class_name not in desired_classes:
                    continue
                [x1, y1, x2, y2] = map(int, box.xyxy[0])
                colour = getColours(cls)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, f'{class_name} {box.conf[0]:.2f}', (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)


    cv2.imshow('frame', frame)
    processed_frames += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

t_proc_end = time.time()
print(f"Processed: {processed_frames} in {t_proc_end - t_proc_start} seconds")

videoCap.release()
cv2.destroyAllWindows()

