import cv2
import numpy as np
from pathlib import Path
from utils import get_parking_spots_bboxes, empty_or_not, get_parking_spots_from_json

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

mask_path = "mask/mask_1920_1080.png"
video_path = "data/parking_1920_1080_loop.mp4"
json_path = None

mask = cv2.imread(mask_path, 0)

cap = cv2.VideoCapture(video_path)

ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read video file.")
    exit()

video_h, video_w = first_frame.shape[:2]


if json_path and Path(json_path).exists():
    print(f"Loading spots from JSON: {json_path}")
    spots = get_parking_spots_from_json(json_path, video_w, video_h)
else:
    print("Loading spots from mask image")
    mask = cv2.resize(mask, (video_w, video_h))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spots = get_parking_spots_bboxes(contours)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

spots_status = [None for _ in spots]
diffs = [None for _ in spots]
previous_frame = None

frame_number = 0
ret = True
step = 30

while True:
    ret, frame = cap.read()

    if not ret:
        break
    if frame_number % step == 0 and previous_frame is not None:
        for spot_idx, spot in enumerate(spots):
            x1, y1, w, h = spot['aabb']
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            diffs[spot_idx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

    if frame_number % step == 0:
        if previous_frame is None:
            array_ = range(len(spots))
        else:
            array_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
        for spot_idx in array_:
            spot = spots[spot_idx]
            x1, y1, w, h = spot['aabb']
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_idx] = spot_status

    if frame_number % step == 0:
        previous_frame = frame.copy()

    for spot_idx, spot in enumerate(spots):
        spot_status = spots_status[spot_idx]
        
        if spot_status:
            cv2.drawContours(frame, [spot['rotated_box']], 0, (0, 255, 0), 2)
        else:
            cv2.drawContours(frame, [spot['rotated_box']], 0, (0, 0, 255), 2)

    cv2.putText(frame, "Available spots: {} / {}".format(str(sum(spots_status)), str(len(spots_status))), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_number += 1

cap.release()
cv2.destroyAllWindows()

