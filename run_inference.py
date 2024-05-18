import cv2
import os
import argparse
import torch
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
from datetime import datetime
import csv
import math

#helper function for timestamping - in progress not working properly
def seconds_to_minutes_and_seconds(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes):02d}:{int(seconds):02d}" 

def pixels_to_feet(altitude, pixel_size, original_frame_width):
    aspect_ratio = 1.7777777777
    fov = (altitude * aspect_ratio * 2 * math.tan(math.radians(73) / 2)) / (np.sqrt(1 + aspect_ratio ** 2))
    size_m = pixel_size / original_frame_width * fov
    return size_m * 3.28084

def get_grade_folder(confidence):
    if confidence > 0.90:
        return 'A_Grade'
    elif confidence > 0.80:
        return 'B_Grade'
    elif confidence > 0.70:
        return 'C_Grade'
    elif confidence > 0.60:
        return 'D_Grade'
    else:
        return 'F_Grade'
    
def save_detected_shark_frame(frame, frame_no_bb, frame_number, track_id, avg_conf, max_conf, min_conf, length, video_fps, video, csv_writer, survey_path, cap):
    grade_folder = get_grade_folder(avg_conf)
    bb_save_path = os.path.join(f'{survey_path}/bb', grade_folder)
    no_bb_save_path = os.path.join(f'{survey_path}/no_bb', grade_folder)

    os.makedirs(bb_save_path, exist_ok=True)
    os.makedirs(no_bb_save_path, exist_ok=True)

    video_file = os.path.basename(video)
    video_name = video_file.split(".")[0]
    frame_bb_filename = os.path.join(bb_save_path, f'{video_name}_shark_{track_id}.jpg')
    frame_no_bb_filename =  os.path.join(no_bb_save_path, f'{video_name}_shark_{track_id}.jpg') 
    
    cv2.imwrite(frame_bb_filename, frame)
    cv2.imwrite(frame_no_bb_filename, frame_no_bb)
    
    seconds = frame_number / video_fps
    timestamp = seconds_to_minutes_and_seconds(seconds)
    
    csv_writer.writerow([video_file, track_id, frame_number, timestamp, avg_conf, max_conf, min_conf, grade_folder.split('_')[0], length])
    print(f"Saved frame for shark {track_id} with confidence {avg_conf:.2f} to {grade_folder} at {timestamp}")

def run_inference(video_directory='survey_video', model_weights_path='model_weights/exp1v8sbest.pt', altitude=30, show_ui=False, years=None):

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    desired_frame_rate = 8 if device != 'cpu' else 4
    
    print(f'Using {device} for inference.')

    if years:
        years = set(map(str, years))
    
    videos = [os.path.join(video_directory, file) for file in os.listdir(video_directory) if not file.startswith('.') and (not years or any(year in file for year in years))]

    results_path = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    survey_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    survey_folder = f'survey_{survey_datetime}'
    survey_path = os.path.join(results_path, survey_folder)
    os.makedirs(survey_path, exist_ok=True)
    
    csv_file = open(f'{survey_path}/shark_detections_{survey_datetime}.csv', mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Video', 'Track ID', 'Frame Number', 'Timestamp', 'Average Confidence', 'Max Confidence', 'Min Confidence', 'Grade', 'Length'])
    
    for video in videos:
        model = YOLO(model_weights_path).to(device=device)

        tracked_sharks = defaultdict(lambda: {"positions": [], "count": 0, "confidences": [], "lengths": [], "frame_number": None, "frame": {}, "frame_no_bb": {}})

        print(f'Processing {video.split("/")[-1]}')
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f'Error opening {video} file')
            continue
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_rate_sample = round(video_fps/desired_frame_rate)
        original_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        frame_number = 1

        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            
            if not success:
                break;

            if (frame_number % frame_rate_sample) == 0:
                resized_frame = cv2.resize(frame, (1280, 720))
                model_frame = resized_frame.copy()
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = model.track(model_frame, persist=True, conf=0.5, device=device, iou=0.25, verbose=False, show=show_ui)

                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu().tolist()
                confidences = results[0].boxes.conf.cpu().tolist()
                track_ids = results[0].boxes.id.cpu().tolist() if results[0].boxes.id is not None else []

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                for box, track_id, confidence in zip(boxes, track_ids, confidences):
                    x, y, w, h = box
                    if track_id not in tracked_sharks:
                        tracked_sharks[track_id] = {"positions": [], "count": 0, "confidences": [], "lengths": [], "frame": {}, "frame_no_bb": {}}
                    track_data = tracked_sharks[track_id]
                    track_data["positions"].append((float(x), float(y)))
                    track_data["count"] += 1
                    track_data["confidences"].append(confidence)
                    track_data["frame_number"] = frame_number
                    track_data["frame"][frame_number] = annotated_frame.copy()
                    track_data["frame_no_bb"][frame_number] = model_frame

                    long_side = max(w, h)
                    short_side = min(w, h)
                    if short_side / long_side >= 0.57:
                        length = pixels_to_feet(altitude, (short_side**2 + long_side**2)**0.5, original_frame_width)
                    else:
                        length = pixels_to_feet(altitude, long_side, original_frame_width)
                    track_data["lengths"].append(length)

                    if len(track_data["positions"]) > 30:  # retain 30 tracks for 30 frames
                        track_data["positions"].pop(0)

                # Display the annotated frame
                if show_ui:
                    cv2.imshow("YOLOv8 Tracking", annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            
            frame_number += 1
            
        for track_id, track_data in tracked_sharks.items():
            if track_data["count"] > 3:
                frame_numbers = sorted(track_data["frame"].keys())
                middle_frame_number = frame_numbers[len(frame_numbers) // 2]
                frame = track_data["frame"][middle_frame_number]
                frame_no_bb = track_data["frame_no_bb"][middle_frame_number]
                avg_confidence = np.mean(track_data["confidences"])
                max_conf = np.max(track_data["confidences"])
                min_conf = np.min(track_data["confidences"])
                avg_length = np.mean(track_data["lengths"])
                save_detected_shark_frame(frame, frame_no_bb, middle_frame_number, track_id, avg_confidence, max_conf, min_conf, avg_length, video_fps, video, csv_writer, survey_path, cap)
        
        cv2.destroyAllWindows()
        cap.release()
    
    csv_file.close()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_directory', type=str, default='survey_video', help='folder where videos to process exist')
    parser.add_argument('--model_weights_path', type=str, default='model_weights/exp1v8sbest.pt', help='path where YoloV8 model exists')
    parser.add_argument('--altitude', type=int, default=40, help='survey flight altitude (meters)')
    parser.add_argument('--show_ui', action='store_true', help='Disable all UI elements during execution')
    parser.add_argument('--years', nargs='*', type=int, help='list of years to filter videos by')
    opt = parser.parse_args()
    return opt

def main(opt):
    run_inference(**vars(opt))

if __name__=='__main__':
    opt = parse_opt()    
    main(opt)



