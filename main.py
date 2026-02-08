import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
Stair Safety Tracker — Core Module
===================================
Tracks people on a staircase via Orbbec camera + YOLO pose detection.
An ESP32 with MPR121 capacitive sensor sends UDP packets when a handrail
segment is touched — the MPR channel number identifies which segment.

Hardware : Orbbec depth camera (color only), ESP32 + MPR121
Protocol : ESP32 sends "TOUCH <channel> <millis>" over UDP
ROI      : Single rectangle drawn around the staircase area
CSV      : Logs each person when they leave the ROI (used_railing, channels, duration)
"""

import time
import socket
import csv
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat

# ── Config ────────────────────────────────────────────────────────────────────
POSE_MODEL    = r"C:\orbbec\venv\object_detection\models\yolo11x-pose.pt"
DEVICE        = 0 if torch.cuda.is_available() else "cpu"
CONF_THRESH   = 0.5
IMG_SIZE      = 640
FRAME_TIMEOUT = 300

UDP_PORT      = 4215
MPR_CHANNELS  = 12
TOUCH_TTL     = 10.0
MAX_DISAPPEARED = 15
CSV_FILE      = "railing_usage.csv"


# ── Orbbec Camera ─────────────────────────────────────────────────────────────
def start_camera():
    pipe, cfg = Pipeline(), Config()
    prof = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR) \
               .get_video_stream_profile(1280, 800, OBFormat.RGB, 30)
    cfg.enable_stream(prof)
    pipe.start(cfg)
    return pipe

def frame_to_bgr(color_frame):
    vf = color_frame.as_video_frame()
    w, h = vf.get_width(), vf.get_height()
    return cv2.cvtColor(
        np.frombuffer(vf.get_data(), np.uint8).reshape(h, w, 3),
        cv2.COLOR_RGB2BGR)


# ── UDP Receiver ──────────────────────────────────────────────────────────────
def udp_setup(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    sock.setblocking(False)
    return sock

def udp_poll(sock):
    """Return list of (channel, timestamp) from pending packets."""
    events = []
    while True:
        try:
            data, _ = sock.recvfrom(1024)
        except BlockingIOError:
            break
        msg = data.decode(errors="ignore").strip()
        print(f"[UDP RAW] '{msg}'")
        parts = msg.split()
        if len(parts) >= 2 and parts[0] == "TOUCH":
            try:
                ch = int(parts[1])
                if 0 <= ch < MPR_CHANNELS:
                    events.append((ch, time.time()))
            except (ValueError, IndexError):
                pass
    return events


# ── Person Tracker (from killerv11 — proven working) ─────────────────────────
class PersonTracker:
    def __init__(self, max_disappeared=MAX_DISAPPEARED):
        self.next_id = 0
        self.objects = {}       # id → {center, box, feet}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid, box, feet):
        self.objects[self.next_id] = {'center': centroid, 'box': box, 'feet': feet}
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        """detections: list of {center, box, feet}"""
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        if len(self.objects) == 0:
            for det in detections:
                self.register(det['center'], det['box'], det['feet'])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[oid]['center'] for oid in object_ids]
            detection_centroids = [d['center'] for d in detections]

            D = np.zeros((len(object_centroids), len(detection_centroids)))
            for i, oc in enumerate(object_centroids):
                for j, dc in enumerate(detection_centroids):
                    D[i, j] = np.linalg.norm(np.array(oc) - np.array(dc))

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > 100:
                    continue
                obj_id = object_ids[row]
                self.objects[obj_id] = detections[col]
                self.disappeared[obj_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            for row in set(range(D.shape[0])) - used_rows:
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

            for col in set(range(D.shape[1])) - used_cols:
                self.register(detections[col]['center'], detections[col]['box'], detections[col]['feet'])

        return self.objects


# ── YOLO Detection (from killerv11 — defensive) ──────────────────────────────
def extract_person_detections(result, roi=None):
    detections = []
    if not hasattr(result, "boxes") or result.boxes is None:
        return detections

    boxes = result.boxes
    xyxy, cls, conf = boxes.xyxy, boxes.cls, boxes.conf
    if xyxy is None or cls is None or conf is None:
        return detections

    if hasattr(xyxy, "cpu"):
        xyxy = xyxy.cpu().numpy()
        cls = cls.cpu().numpy()
        conf = conf.cpu().numpy()
    else:
        xyxy = np.asarray(xyxy)
        cls = np.asarray(cls)
        conf = np.asarray(conf)

    keypoints = None
    if hasattr(result, "keypoints") and result.keypoints is not None:
        kp = result.keypoints
        if hasattr(kp, "xy"):
            keypoints = kp.xy
            if hasattr(keypoints, "cpu"):
                keypoints = keypoints.cpu().numpy()
            else:
                keypoints = np.asarray(keypoints)

    for i in range(len(xyxy)):
        if conf[i] < CONF_THRESH or int(cls[i]) != 0:
            continue
        x1, y1, x2, y2 = xyxy[i]
        cx, cy = int(0.5 * (x1 + x2)), int(0.5 * (y1 + y2))

        # ROI filter
        if roi and not (roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]):
            continue

        feet = []
        if keypoints is not None and i < len(keypoints):
            kp_person = keypoints[i]
            for idx in [15, 16]:  # COCO: L/R ankle
                if idx < len(kp_person):
                    kx, ky = kp_person[idx][:2]
                    if kx > 0 and ky > 0:
                        feet.append((int(kx), int(ky)))
        if not feet:
            feet = [(cx, int(y2))]

        detections.append({
            'center': (cx, cy),
            'box': (int(x1), int(y1), int(x2), int(y2)),
            'feet': feet
        })
    return detections


# ── Touch Assigner ────────────────────────────────────────────────────────────
class TouchAssigner:
    def __init__(self):
        self.assignments = defaultdict(list)  # pid → [(channel, time)]

    def assign(self, channel, persons, now):
        """Assign touch to the person closest to bottom of frame (lowest on stairs)."""
        if not persons:
            print(f"[TOUCH] ch{channel} received but no persons in ROI")
            return None
        pid = max(persons, key=lambda p: persons[p]["center"][1])
        self.assignments[pid].append((channel, now))
        print(f"[TOUCH] ch{channel} → Person {pid}")
        return pid

    def active(self, now):
        """Return {pid: [channels]} for recent touches."""
        out = {}
        for pid, entries in list(self.assignments.items()):
            live = [ch for ch, t in entries if now - t < TOUCH_TTL]
            if live:
                out[pid] = live
            else:
                del self.assignments[pid]
        return out

    def has_ever_touched(self, person_id):
        return person_id in self.assignments and len(self.assignments[person_id]) > 0

    def get_channels_touched(self, person_id):
        if person_id not in self.assignments:
            return []
        return sorted(set(ch for ch, ts in self.assignments[person_id]))


# ── CSV Logger (from killerv11 — logs on person disappearance) ───────────────
class CSVLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self.logged_persons = set()
        self.person_first_seen = {}

        if not os.path.isfile(filepath):
            with open(filepath, 'w', newline='') as f:
                csv.writer(f).writerow([
                    'timestamp', 'person_id', 'used_railing',
                    'channels_touched', 'duration_seconds'
                ])
            print(f"[CSV] Created: {filepath}")

    def update(self, persons, assigner, now):
        """Call every frame. Tracks enter/exit and logs on disappearance."""
        current_ids = set(persons.keys())

        # Track new persons
        for pid in current_ids:
            if pid not in self.person_first_seen:
                self.person_first_seen[pid] = now
                print(f"[CSV] Person {pid} entered")

        # Detect disappeared and log them
        previously_seen = set(self.person_first_seen.keys())
        for pid in previously_seen - current_ids:
            if pid not in self.logged_persons:
                self._log_person(pid, assigner, now)
                self.logged_persons.add(pid)
                if pid in self.person_first_seen:
                    del self.person_first_seen[pid]

    def _log_person(self, pid, assigner, now):
        used = assigner.has_ever_touched(pid)
        channels = assigner.get_channels_touched(pid)
        ch_str = ",".join(map(str, channels)) if channels else "none"
        duration = now - self.person_first_seen.get(pid, now)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.filepath, 'a', newline='') as f:
            csv.writer(f).writerow([
                ts, pid, 'YES' if used else 'NO', ch_str, f"{duration:.1f}"
            ])
        status = "✓ USED railing" if used else "✗ NO railing"
        print(f"[CSV] Logged Person {pid}: {status} (ch: {ch_str}, {duration:.1f}s)")

    def force_log_all(self, persons, assigner, now):
        """Log all current persons (e.g. on shutdown/reset)."""
        for pid in persons.keys():
            if pid not in self.logged_persons:
                self._log_person(pid, assigner, now)
                self.logged_persons.add(pid)


# ── ROI Setup (single rectangle) ─────────────────────────────────────────────
roi = None
_drag = {"on": False, "a": (0, 0), "b": (0, 0)}

def _mouse_cb(event, x, y, flags, param):
    global roi
    if event == cv2.EVENT_LBUTTONDOWN:
        _drag.update(on=True, a=(x, y), b=(x, y))
    elif event == cv2.EVENT_MOUSEMOVE and _drag["on"]:
        _drag["b"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        _drag["on"] = False; _drag["b"] = (x, y)
        ax, ay = _drag["a"]; bx, by = _drag["b"]
        roi = (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))
        print(f"[ROI] {roi}")


# ── Drawing ───────────────────────────────────────────────────────────────────
COLORS = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(128,0,128),(255,128,0)]

def draw(img, tracker, assigner, mode):
    out = img.copy()
    if mode == "setup":
        if _drag["on"]:
            cv2.rectangle(out, _drag["a"], _drag["b"], (255, 255, 0), 3)
        if roi:
            cv2.rectangle(out, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
        msg = "ROI set — press C" if roi else "Draw ROI around staircase"
        cv2.putText(out, msg, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        if roi:
            cv2.rectangle(out, (roi[0], roi[1]), (roi[2], roi[3]), (60, 60, 60), 1)
        now = time.time()
        touches = assigner.active(now)
        for pid, p in tracker.objects.items():
            col = COLORS[pid % len(COLORS)]
            x1, y1, x2, y2 = p["box"]
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 3)
            cv2.putText(out, f"Person {pid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            for ft in p["feet"]:
                cv2.circle(out, ft, 6, col, -1)
            if pid in touches:
                ch_str = ",".join(str(c) for c in touches[pid])
                cv2.putText(out, f"railing: ch{ch_str}", (x1, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        cv2.putText(out, f"Active persons: {len(tracker.objects)}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(out, "ESC = Quit | R = Reset", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global roi
    print("[INFO] Loading YOLO:", POSE_MODEL)
    model = YOLO(POSE_MODEL)
    try: model.to(DEVICE)
    except: pass

    pipe = start_camera()
    sock = udp_setup(UDP_PORT)
    tracker  = PersonTracker()
    assigner = TouchAssigner()
    logger   = CSVLogger(CSV_FILE)
    mode = "setup"

    win = "StairTracker"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 800)
    cv2.setMouseCallback(win, _mouse_cb)

    print("SETUP: draw ROI around staircase, press C to start")
    print(f"ESP32 UDP port {UDP_PORT} | Logging to {CSV_FILE}")
    print("ESC=quit  R=reset  C=confirm ROI")

    try:
        while True:
            fs = pipe.wait_for_frames(FRAME_TIMEOUT)
            if fs is None: continue
            cf = fs.as_frame_set().get_color_frame()
            if cf is None: continue
            bgr = frame_to_bgr(cf)

            if mode == "tracking":
                res = model.track(
                    source=bgr[:, :, ::-1],
                    device=DEVICE, verbose=False,
                    conf=CONF_THRESH, imgsz=IMG_SIZE,
                    persist=True, tracker="bytetrack.yaml"
                )[0]

                detections = extract_person_detections(res, roi)
                tracker.update(detections)

                # Poll ESP32 touches
                for ch, ts in udp_poll(sock):
                    assigner.assign(ch, tracker.objects, ts)

                # CSV: track enter/exit
                logger.update(tracker.objects, assigner, time.time())

            cv2.imshow(win, draw(bgr, tracker, assigner, mode))

            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            elif key in (ord('r'), ord('R')):
                if mode == "tracking":
                    logger.force_log_all(tracker.objects, assigner, time.time())
                roi = None; mode = "setup"
                tracker  = PersonTracker()
                assigner = TouchAssigner()
                logger   = CSVLogger(CSV_FILE)
                print("[RESET] Back to setup")
            elif key in (ord('c'), ord('C')) and mode == "setup" and roi:
                mode = "tracking"
                print("[TRACKING] Started")
    finally:
        if mode == "tracking":
            logger.force_log_all(tracker.objects, assigner, time.time())
            print(f"[CSV] Final log → {CSV_FILE}")
        try: pipe.stop()
        except: pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()