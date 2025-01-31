import multiprocessing
import cv2
import time
import random
import threading
import concurrent.futures
from queue import Queue
from ultralytics import YOLO

_model = None

def init_inference_worker():
    global _model
    _model = YOLO('yolov8n.pt')

def inference_worker(frame, selected_classes, all_classes):
    global _model
    results = _model.predict(frame, stream=True)
    detections = []
    for result in results:
        classes_names = result.names
        for box in result.boxes:
            if box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                class_name = classes_names[cls]
                if not all_classes:
                    if class_name in selected_classes:
                        detections.append((x1, y1, x2, y2, class_name, float(box.conf[0])))
                else:
                    detections.append((x1, y1, x2, y2, class_name, float(box.conf[0])))
    return frame, detections

def generate_class_colors(class_names):
    random.seed(75)
    return {name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for name in class_names}

def read_frames(cap, frame_queue, stop_event):
    fps = cap.get(cv2.CAP_PROP_FPS)
    sleep_time = 1 / fps if fps > 0 else 0.03
    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            stop_event.set()
            break
        frame_queue.put(frame)
        time.sleep(sleep_time)
    cap.release()

def display_frames(display_queue, class_colors, stop_event):
    frames_count = 0
    start_time = time.time()
    while not stop_event.is_set():
        try:
            frame, detections = display_queue.get(timeout=0.1)
        except:
            continue
        frames_count += 1
        for x1, y1, x2, y2, class_name, conf in detections:
            color = class_colors.get(class_name, (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break
    end_time = time.time()
    print(f"Processed Frames: {frames_count} in {end_time - start_time} seconds")
    cv2.destroyAllWindows()

def main():
    video_path = "pato_to_video"
    selected_classes = ['car', 'person']
    all_classes = True
    class_colors = generate_class_colors(selected_classes)

    cap = cv2.VideoCapture(video_path if video_path else 0)
    if not cap.isOpened():
        raise ValueError("ERROR: Video is not readable. Make sure path is correct and try again.")

    frame_queue = Queue(maxsize=10)
    display_queue = Queue()
    stop_event = threading.Event()

    reader_thread = threading.Thread(target=read_frames, args=(cap, frame_queue, stop_event))
    reader_thread.start()

    display_thread = threading.Thread(target=display_frames, args=(display_queue, class_colors, stop_event))
    display_thread.start()

    with concurrent.futures.ProcessPoolExecutor(max_workers=multiprocessing.cpu_count(), initializer=init_inference_worker) as executor:
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.1)
            except:
                continue
            future = executor.submit(inference_worker, frame, selected_classes, all_classes)
            try:
                frame_result, detections = future.result()
            except:
                continue
            display_queue.put((frame_result, detections))

    reader_thread.join()
    display_thread.join()

if __name__ == "__main__":
    main()