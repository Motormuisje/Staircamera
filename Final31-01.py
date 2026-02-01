"""
STAIR SAFETY TRACKER v4 - Multi-Mode Fall Detection
====================================================

Features:
- Multiple fall detection methods (toggle with F1-F7)
- 3D depth-based measurements using Orbbec camera
- Floor plane ROI definition for stair area
- Real-time visualization of all keypoints and metrics
- Long beep alarm on fall detection

Fall Detection Methods:
1. VELOCITY    - Sudden vertical speed increase
2. TORSO_ANGLE - Angle between torso and floor
3. TORSO_FLOOR - Distance from torso to floor plane
4. HAND_POSITION - Hands below normal position (catching fall)
5. HAND_FEET_OVERLAP - Hands near feet level (fallen posture)
6. BODY_RATIO  - Height/width ratio change
7. COMBINED    - Weighted combination of all methods

Author: Safety Engineering Project
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import socket
import platform
import re
import math
import threading
from collections import defaultdict, deque
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum, auto
import csv

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat


# ===================== CONFIGURATION =====================
@dataclass
class AppConfig:
    """Application configuration"""
    # Model
    pose_checkpoint: str = r"C:\orbbec\venv\object_detection\models\yolo11x-pose.pt"
    confidence: float = 0.5
    image_size: int = 640
    
    # Camera
    color_width: int = 1280
    color_height: int = 800
    depth_width: int = 640
    depth_height: int = 400
    camera_timeout_ms: int = 300
    
    # Camera intrinsics (Orbbec)
    fx: float = 610.97
    fy: float = 611.22
    cx: float = 628.07
    cy: float = 356.80
    
    # UDP
    udp_port: int = 4213
    num_segments: int = 4
    
    # Tracking
    max_disappeared_frames: int = 15
    max_assign_distance_px: int = 500
    
    # Output
    csv_output: str = "railing_usage.csv"
    fall_csv_output: str = "falls.csv"


APP_CONFIG = AppConfig()
DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ===================== AUDIO FEEDBACK =====================
if platform.system() == "Windows":
    import winsound
    def beep_touch(): winsound.Beep(1400, 100)
    def beep_warning(): winsound.Beep(600, 300)
    def beep_fall_alarm(): 
        # Lange beep in aparte thread zodat het niet blokkeert
        winsound.Beep(800, 2000)
else:
    def beep_touch(): pass
    def beep_warning(): pass
    def beep_fall_alarm(): pass


# ===================== FALL DETECTION MODES =====================
class FallDetectionMode(Enum):
    """Available fall detection methods"""
    VELOCITY = auto()         # F1 - Vertical velocity spike
    TORSO_ANGLE = auto()      # F2 - Torso angle from vertical
    LEG_ANGLE = auto()        # F3 - Knee-ankle angle (horizontal leg = fallen)
    SIMPLE_OR = auto()        # F4 - Velocity OR Torso OR Leg (simple, reliable)
    ACCELERATION = auto()     # F5 - How fast velocity increases
    VELOCITY_AND_ANGLE = auto() # F6 - Both velocity AND angle (strictest)
    TORSO_FLOOR = auto()      # F7 - Torso distance to floor (3D)
    HAND_POSITION = auto()    # F8 - Hands lower than normal
    HAND_FEET_OVERLAP = auto() # F9 - Hands near feet level
    BODY_RATIO = auto()       # F10 - Body aspect ratio change
    COMBINED = auto()         # F11 - Weighted combination

    @classmethod
    def get_name(cls, mode):
        names = {
            cls.VELOCITY: "VELOCITY (snelheid)",
            cls.TORSO_ANGLE: "TORSO_ANGLE (hoek torso)",
            cls.LEG_ANGLE: "LEG_ANGLE (hoek been)",
            cls.SIMPLE_OR: "SIMPLE_OR (vel OF torso OF been)",
            cls.ACCELERATION: "ACCELERATION (versnelling)",
            cls.VELOCITY_AND_ANGLE: "VELOCITY+ANGLE (beide)",
            cls.TORSO_FLOOR: "TORSO_FLOOR (3D afstand)",
            cls.HAND_POSITION: "HAND_POSITION (handen laag)",
            cls.HAND_FEET_OVERLAP: "HAND_FEET_OVERLAP (handen bij voeten)",
            cls.BODY_RATIO: "BODY_RATIO (lichaamsverhouding)",
            cls.COMBINED: "COMBINED (alle methodes)"
        }
        return names.get(mode, str(mode))


@dataclass
class FallConfig:
    """Configuration for fall detection thresholds"""
    # Minimum frames before detection starts
    min_tracking_frames: int = 10
    calibration_frames: int = 15
    
    # VELOCITY thresholds
    velocity_thresh_px_per_frame: float = 8.0  # pixels/frame vertical speed
    
    # ACCELERATION thresholds (how fast velocity changes)
    acceleration_thresh: float = 3.0  # px/frame^2
    
    # TORSO_ANGLE thresholds  
    torso_angle_thresh_deg: float = 55.0  # degrees from vertical
    
    # LEG_ANGLE thresholds (knee-to-ankle angle)
    # Normal standing: ~80-90° from horizontal (vertical leg)
    # Fallen/horizontal leg: < 30° from horizontal
    leg_angle_thresh_deg: float = 35.0  # degrees from horizontal (< this = fallen)
    
    # TORSO_FLOOR thresholds (3D depth)
    torso_floor_thresh_mm: float = 600.0  # mm above floor
    torso_floor_ratio: float = 0.45  # ratio vs standing height
    
    # HAND_POSITION thresholds
    hand_drop_ratio: float = 0.3  # hands dropped 30% of body height
    
    # HAND_FEET_OVERLAP thresholds
    hand_feet_dist_ratio: float = 0.2  # hands within 20% of body height from feet
    
    # BODY_RATIO thresholds
    aspect_ratio_thresh: float = 0.85  # width/height > 0.85 = horizontal
    height_collapse_ratio: float = 0.5  # height < 50% of standing = collapsed
    
    # Confirmation
    persist_frames: int = 8  # Reduced for faster detection
    cooldown_sec: float = 10.0
    
    # Combined mode weights
    weight_velocity: float = 0.15
    weight_torso_angle: float = 0.20
    weight_acceleration: float = 0.15
    weight_torso_floor: float = 0.20
    weight_hand_position: float = 0.10
    weight_hand_feet: float = 0.10
    weight_body_ratio: float = 0.10


FALL_CONFIG = FallConfig()


# ===================== SAFETY FACTS (voor feedback) =====================
SAFETY_FACTS = [
    "Wist je dat? Elke 5 minuten belandt een 65-plusser op de SEH door een val!",
    "Wist je dat? In Nederland overlijden dagelijks 19 mensen door een val.",
    "Wist je dat? Een val van de trap is 6x dodelijker dan een verkeersongeval!",
    "Wist je dat? 112.000 ouderen per jaar op de SEH belanden door een val.",
    "Wist je dat? 19% van trapvallen resulteert in blijvend letsel.",
    "Wist je dat? 80% van alle dodelijke ongevallen in NL zijn valpartijen!",
    "Wist je dat? Bij 76% van de vallen gaat het om een vaste trap zoals deze.",
]


# ===================== COCO KEYPOINT INDICES =====================
class KeypointIdx:
    """COCO pose keypoint indices"""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


# ===================== DATA CLASSES =====================
@dataclass
class KeypointData:
    """All extracted keypoints for a person"""
    raw: np.ndarray  # Full (17, 2) or (17, 3) array
    
    # Head
    nose: Optional[Tuple[int, int]] = None
    
    # Upper body
    left_shoulder: Optional[Tuple[int, int]] = None
    right_shoulder: Optional[Tuple[int, int]] = None
    left_elbow: Optional[Tuple[int, int]] = None
    right_elbow: Optional[Tuple[int, int]] = None
    left_wrist: Optional[Tuple[int, int]] = None
    right_wrist: Optional[Tuple[int, int]] = None
    
    # Lower body
    left_hip: Optional[Tuple[int, int]] = None
    right_hip: Optional[Tuple[int, int]] = None
    left_knee: Optional[Tuple[int, int]] = None
    right_knee: Optional[Tuple[int, int]] = None
    left_ankle: Optional[Tuple[int, int]] = None
    right_ankle: Optional[Tuple[int, int]] = None
    
    @property
    def shoulder_mid(self) -> Optional[Tuple[int, int]]:
        if self.left_shoulder and self.right_shoulder:
            return ((self.left_shoulder[0] + self.right_shoulder[0]) // 2,
                    (self.left_shoulder[1] + self.right_shoulder[1]) // 2)
        return self.left_shoulder or self.right_shoulder
    
    @property
    def hip_mid(self) -> Optional[Tuple[int, int]]:
        if self.left_hip and self.right_hip:
            return ((self.left_hip[0] + self.right_hip[0]) // 2,
                    (self.left_hip[1] + self.right_hip[1]) // 2)
        return self.left_hip or self.right_hip
    
    @property
    def hand_mid(self) -> Optional[Tuple[int, int]]:
        hands = [h for h in [self.left_wrist, self.right_wrist] if h]
        if hands:
            return (sum(h[0] for h in hands) // len(hands),
                    sum(h[1] for h in hands) // len(hands))
        return None
    
    @property
    def feet_mid(self) -> Optional[Tuple[int, int]]:
        feet = [f for f in [self.left_ankle, self.right_ankle] if f]
        if feet:
            return (sum(f[0] for f in feet) // len(feet),
                    sum(f[1] for f in feet) // len(feet))
        return None


@dataclass
class PersonData:
    """Complete tracking data for a single person"""
    center: Tuple[int, int]
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    feet: List[Tuple[int, int]]
    keypoints: Optional[KeypointData] = None


@dataclass
class PersonFallState:
    """Fall detection state for a single person"""
    person_id: int
    
    # Calibration (reference values when standing)
    is_calibrated: bool = False
    calibration_frames: int = 0
    
    # Reference measurements
    ref_torso_y: Optional[float] = None  # Y position of torso when standing
    ref_height: Optional[float] = None  # Bbox height when standing
    ref_hand_y: Optional[float] = None  # Hand Y position when standing
    ref_torso_depth: Optional[float] = None  # 3D depth of torso when standing
    
    # History for velocity calculation
    center_history: List[Tuple[int, int]] = field(default_factory=list)
    
    # Current measurements
    current_velocity: float = 0.0
    current_velocity_compensated: float = 0.0  # Depth-compensated velocity
    current_depth: Optional[float] = None  # Current torso depth
    current_acceleration: float = 0.0
    current_torso_angle: float = 0.0
    current_leg_angle: float = 90.0  # Knee-ankle angle from horizontal (90 = vertical leg)
    current_torso_floor_dist: float = 0.0
    current_hand_drop: float = 0.0
    current_hand_feet_dist: float = 0.0
    current_aspect_ratio: float = 0.0
    current_height_ratio: float = 1.0
    
    # Velocity history for acceleration calculation
    velocity_history: List[float] = field(default_factory=list)
    
    # Detection state
    method_scores: Dict[FallDetectionMode, float] = field(default_factory=dict)
    consecutive_fall_frames: int = 0
    is_fallen: bool = False
    fall_detected_at: Optional[float] = None
    last_alert_time: float = 0.0
    triggered_by: Optional[FallDetectionMode] = None


# ===================== FLOOR PLANE =====================
@dataclass
class FloorPlane:
    """
    Represents the floor/stair surface as a plane in 3D space.
    Defined by 3 or 4 points clicked by user.
    """
    points_2d: List[Tuple[int, int]] = field(default_factory=list)  # Pixel coordinates
    points_3d: List[Tuple[float, float, float]] = field(default_factory=list)  # 3D coordinates
    
    # Plane equation: ax + by + cz + d = 0
    normal: Optional[np.ndarray] = None  # (a, b, c)
    d: float = 0.0
    
    is_defined: bool = False
    
    def add_point(self, px: int, py: int, depth_mm: float, config: AppConfig):
        """Add a point to define the floor plane"""
        if len(self.points_2d) >= 4:
            return
        
        # Convert 2D + depth to 3D
        x3d, y3d, z3d = pixel_to_3d(px, py, depth_mm, config)
        
        self.points_2d.append((px, py))
        self.points_3d.append((x3d, y3d, z3d))
        
        print(f"[FLOOR] Point {len(self.points_2d)}: pixel=({px},{py}), depth={depth_mm:.0f}mm, 3D=({x3d:.0f},{y3d:.0f},{z3d:.0f})")
        
        if len(self.points_3d) >= 3:
            self._compute_plane()
    
    def _compute_plane(self):
        """Compute plane equation from 3+ points"""
        if len(self.points_3d) < 3:
            return
        
        p1 = np.array(self.points_3d[0])
        p2 = np.array(self.points_3d[1])
        p3 = np.array(self.points_3d[2])
        
        # Two vectors in the plane
        v1 = p2 - p1
        v2 = p3 - p1
        
        # Normal = cross product
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            print("[FLOOR] Warning: Points are collinear, cannot define plane")
            return
        
        self.normal = normal / norm
        self.d = -np.dot(self.normal, p1)
        self.is_defined = True
        
        print(f"[FLOOR] Plane defined: normal=({self.normal[0]:.3f},{self.normal[1]:.3f},{self.normal[2]:.3f}), d={self.d:.1f}")
    
    def distance_to_point(self, x3d: float, y3d: float, z3d: float) -> float:
        """Calculate signed distance from point to plane (mm)"""
        if not self.is_defined:
            return float('inf')
        
        point = np.array([x3d, y3d, z3d])
        return abs(np.dot(self.normal, point) + self.d)
    
    def reset(self):
        """Reset floor plane"""
        self.points_2d.clear()
        self.points_3d.clear()
        self.normal = None
        self.d = 0.0
        self.is_defined = False


def pixel_to_3d(px: int, py: int, depth_mm: float, config: AppConfig) -> Tuple[float, float, float]:
    """
    Convert pixel coordinates + depth to 3D world coordinates.
    Uses pinhole camera model.
    """
    # Scale pixel from color to depth resolution
    px_depth = px * config.depth_width / config.color_width
    py_depth = py * config.depth_height / config.color_height
    
    # Deproject using intrinsics
    z = depth_mm
    x = (px_depth - config.cx) * z / config.fx
    y = (py_depth - config.cy) * z / config.fy
    
    return (x, y, z)


# ===================== UDP RECEIVER =====================
class UDPReceiver:
    """Receives touch events from ESP32"""
    def __init__(self, port: int, num_segments: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.sock.setblocking(False)
        self.num_segments = num_segments
        self._pattern = re.compile(r'^(TOUCH|RELEASE)\s+(\d+)\s+(\d+)$')
    
    def poll(self) -> List[Tuple[int, str, float]]:
        """Poll for events. Returns list of (segment_id, event_type, timestamp)"""
        events = []
        while True:
            try:
                data, _ = self.sock.recvfrom(1024)
            except BlockingIOError:
                break
            
            msg = data.decode(errors="ignore").strip()
            match = self._pattern.match(msg)
            if match:
                event_type = match.group(1).lower()
                seg_id = int(match.group(2))
                if 0 <= seg_id < self.num_segments:
                    events.append((seg_id, event_type, time.time()))
                    if event_type == "touch":
                        beep_touch()
        return events
    
    def close(self):
        self.sock.close()


# ===================== ZONE MANAGER =====================
class ZoneManager:
    """Manages segment zones and stair ROI"""
    def __init__(self, num_segments: int):
        self.num_segments = num_segments
        self.segment_zones: List[Optional[Tuple[int,int,int,int]]] = [None] * num_segments
        self.stair_roi: Optional[Tuple[int,int,int,int]] = None
        
        # Drawing state
        self.current_segment = 0
        self.drawing = False
        self.draw_start = (0, 0)
        self.draw_end = (0, 0)
        self.draw_type = "segment"  # "segment" or "roi"
    
    @property
    def draw_preview(self):
        if self.drawing:
            rect = self._norm_rect(self.draw_start, self.draw_end)
            return (self.draw_type, rect)
        return None
    
    def _norm_rect(self, p1, p2):
        x1, x2 = sorted([int(p1[0]), int(p2[0])])
        y1, y2 = sorted([int(p1[1]), int(p2[1])])
        return (x1, y1, x2, y2)
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.draw_start = (x, y)
            self.draw_end = (x, y)
            self.draw_type = "segment"
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True
            self.draw_start = (x, y)
            self.draw_end = (x, y)
            self.draw_type = "roi"
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.draw_end = (x, y)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            if self.drawing:
                self.drawing = False
                rect = self._norm_rect(self.draw_start, self.draw_end)
                if self.draw_type == "roi":
                    self.stair_roi = rect
                    print(f"[ZONE] Stair ROI defined: {rect}")
                else:
                    self.segment_zones[self.current_segment] = rect
                    print(f"[ZONE] Segment {self.current_segment} defined: {rect}")
    
    def all_defined(self) -> bool:
        return all(z is not None for z in self.segment_zones) and self.stair_roi is not None
    
    def reset(self):
        self.segment_zones = [None] * self.num_segments
        self.stair_roi = None
        self.current_segment = 0
    
    def point_in_roi(self, x: int, y: int) -> bool:
        if self.stair_roi is None:
            return False
        x1, y1, x2, y2 = self.stair_roi
        return x1 <= x <= x2 and y1 <= y <= y2


# ===================== TOUCH TRACKING =====================
class ActiveTouchState:
    """Tracks which segments are currently being touched"""
    def __init__(self):
        self._active: Dict[int, float] = {}  # segment_id -> touch_start_time
    
    def process_event(self, seg_id: int, event_type: str, timestamp: float):
        """Process a touch or release event"""
        if event_type == "touch":
            self._active[seg_id] = timestamp
        elif event_type == "release":
            self._active.pop(seg_id, None)
    
    def is_touched(self, seg_id: int) -> bool:
        return seg_id in self._active
    
    @property
    def active_segments(self) -> Set[int]:
        return set(self._active.keys())


class TouchAssigner:
    """Assigns touch events to persons based on proximity"""
    def __init__(self, zone_manager: ZoneManager, max_distance: int = 500):
        self.zone_manager = zone_manager
        self.max_distance = max_distance
        self._person_segments: Dict[int, Set[int]] = defaultdict(set)
        self._unassigned: List[Tuple[int, float]] = []  # (segment_id, timestamp)
    
    def assign(self, seg_id: int, event_type: str, tracker: 'PersonTracker', timestamp: float):
        """Assign a touch event to nearest person"""
        if event_type != "touch":
            return
        
        zone = self.zone_manager.segment_zones[seg_id]
        if zone is None:
            return
        
        zone_center = ((zone[0] + zone[2]) // 2, (zone[1] + zone[3]) // 2)
        
        best_pid = None
        best_dist = float('inf')
        
        for pid, pdata in tracker.persons.items():
            for foot in pdata.feet:
                dist = np.sqrt((foot[0] - zone_center[0])**2 + (foot[1] - zone_center[1])**2)
                if dist < best_dist:
                    best_dist = dist
                    best_pid = pid
        
        if best_pid is not None and best_dist < self.max_distance:
            self._person_segments[best_pid].add(seg_id)
            print(f"[TOUCH] ✓ Segment {seg_id} → Person {best_pid} ({best_dist:.0f}px)")
        else:
            self._unassigned.append((seg_id, timestamp))
            print(f"[TOUCH] Segment {seg_id} touched - NO PERSON nearby")
    
    def try_assign_unassigned(self, tracker: 'PersonTracker'):
        """Try to assign previously unassigned touches"""
        now = time.time()
        still_unassigned = []
        
        for seg_id, ts in self._unassigned:
            if now - ts > 5.0:  # Expire after 5 seconds
                continue
            
            zone = self.zone_manager.segment_zones[seg_id]
            if zone is None:
                continue
            
            zone_center = ((zone[0] + zone[2]) // 2, (zone[1] + zone[3]) // 2)
            
            best_pid = None
            best_dist = float('inf')
            
            for pid, pdata in tracker.persons.items():
                for foot in pdata.feet:
                    dist = np.sqrt((foot[0] - zone_center[0])**2 + (foot[1] - zone_center[1])**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_pid = pid
            
            if best_pid is not None and best_dist < self.max_distance:
                self._person_segments[best_pid].add(seg_id)
                print(f"[TOUCH] ✓ Late assign: Segment {seg_id} → Person {best_pid}")
            else:
                still_unassigned.append((seg_id, ts))
        
        self._unassigned = still_unassigned
    
    def get_person_segments(self, pid: int) -> Set[int]:
        return self._person_segments.get(pid, set())
    
    def clear_person(self, pid: int):
        self._person_segments.pop(pid, None)


class RailingUsageLogger:
    """Logs railing usage to CSV with daily statistics"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._logged_persons: Set[int] = set()
        self._person_entry_times: Dict[int, float] = {}
        self._safe_users_today = 0
        self._total_users_today = 0
        
        # Create/load CSV
        if not os.path.isfile(filepath):
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'person_id', 'used_railing', 
                               'segments_touched', 'num_segments', 'duration_seconds'])
            print(f"[LOG] Created: {filepath}")
        else:
            self._load_today_stats()
    
    def _load_today_stats(self):
        """Load today's statistics from existing CSV"""
        today = datetime.now().strftime("%Y-%m-%d")
        try:
            with open(self.filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['timestamp'].startswith(today):
                        self._total_users_today += 1
                        if row['used_railing'] == 'YES':
                            num_seg = int(row.get('num_segments', 0))
                            if num_seg >= 2:
                                self._safe_users_today += 1
        except Exception:
            pass
    
    def track_roi_entry(self, pid: int, now: float):
        if pid not in self._person_entry_times:
            self._person_entry_times[pid] = now
    
    def get_entry_time(self, pid: int) -> Optional[float]:
        return self._person_entry_times.get(pid)
    
    def log_person(self, pid: int, touch_assigner: TouchAssigner, now: float):
        if pid in self._logged_persons:
            return
        
        segments = touch_assigner.get_person_segments(pid)
        num_segments = len(segments)
        used_railing = num_segments > 0
        
        entry_time = self._person_entry_times.get(pid, now)
        duration = now - entry_time
        
        segments_str = ",".join(str(s) for s in sorted(segments)) if segments else "none"
        
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                pid,
                'YES' if used_railing else 'NO',
                segments_str,
                num_segments,
                f"{duration:.1f}"
            ])
        
        self._logged_persons.add(pid)
        self._total_users_today += 1
        if num_segments >= 2:
            self._safe_users_today += 1
        
        status = f"✓ {num_segments} segments" if used_railing else "✗ NO railing"
        print(f"[LOG] Person {pid}: {status} (duration: {duration:.1f}s)")
        print(f"[STATS] Safe users today: {self._safe_users_today}/{self._total_users_today}")
    
    def get_safe_users_today(self) -> int:
        return self._safe_users_today
    
    def get_total_users_today(self) -> int:
        return self._total_users_today


# ===================== PERSON TRACKER =====================
class PersonTracker:
    """Centroid-based person tracker"""
    def __init__(self, max_disappeared: int = 15):
        self.next_id = 0
        self.persons: Dict[int, PersonData] = {}
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self._entered_roi: Set[int] = set()
        self._left_roi: Set[int] = set()
    
    def update(self, detections: List[PersonData]):
        self._left_roi.clear()
        
        if len(detections) == 0:
            for pid in list(self.disappeared.keys()):
                self.disappeared[pid] += 1
                if self.disappeared[pid] > self.max_disappeared:
                    if pid in self._entered_roi:
                        self._left_roi.add(pid)
                        self._entered_roi.discard(pid)
                    del self.persons[pid]
                    del self.disappeared[pid]
            return
        
        if len(self.persons) == 0:
            for det in detections:
                self._register(det)
        else:
            self._match_detections(detections)
    
    def _register(self, det: PersonData):
        self.persons[self.next_id] = det
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    
    def _match_detections(self, detections: List[PersonData]):
        object_ids = list(self.persons.keys())
        object_centers = [self.persons[oid].center for oid in object_ids]
        det_centers = [d.center for d in detections]
        
        D = np.zeros((len(object_centers), len(det_centers)))
        for i, oc in enumerate(object_centers):
            for j, dc in enumerate(det_centers):
                D[i, j] = np.linalg.norm(np.array(oc) - np.array(dc))
        
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        
        used_rows, used_cols = set(), set()
        
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > 150:
                continue
            
            oid = object_ids[row]
            self.persons[oid] = detections[col]
            self.disappeared[oid] = 0
            used_rows.add(row)
            used_cols.add(col)
        
        for row in set(range(len(object_ids))) - used_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                if oid in self._entered_roi:
                    self._left_roi.add(oid)
                    self._entered_roi.discard(oid)
                del self.persons[oid]
                del self.disappeared[oid]
        
        for col in set(range(len(detections))) - used_cols:
            self._register(detections[col])
    
    def check_roi_entry(self, zone_manager: ZoneManager):
        for pid, pdata in self.persons.items():
            if pid not in self._entered_roi:
                for foot in pdata.feet:
                    if zone_manager.point_in_roi(foot[0], foot[1]):
                        self._entered_roi.add(pid)
                        break
    
    def has_entered_roi(self, pid: int) -> bool:
        return pid in self._entered_roi
    
    def get_left_roi_persons(self) -> Set[int]:
        return self._left_roi.copy()


# ===================== FALL DETECTOR =====================
class FallDetector:
    """
    Multi-mode fall detection system.
    Supports multiple detection methods that can be toggled.
    """
    
    def __init__(self, config: FallConfig = None):
        self.config = config or FALL_CONFIG
        self.active_mode: FallDetectionMode = FallDetectionMode.COMBINED
        self.person_states: Dict[int, PersonFallState] = {}
        self._fall_events: List[Dict] = []
        
        # Depth data
        self._depth_frame: Optional[np.ndarray] = None
        self._floor_plane: Optional[FloorPlane] = None
    
    def set_mode(self, mode: FallDetectionMode):
        """Set active detection mode"""
        self.active_mode = mode
        print(f"[FALL] Detection mode: {FallDetectionMode.get_name(mode)}")
    
    def cycle_mode(self):
        """Cycle to next detection mode"""
        modes = list(FallDetectionMode)
        idx = modes.index(self.active_mode)
        self.active_mode = modes[(idx + 1) % len(modes)]
        print(f"[FALL] Detection mode: {FallDetectionMode.get_name(self.active_mode)}")
    
    def set_depth_frame(self, depth_frame: Optional[np.ndarray]):
        self._depth_frame = depth_frame
    
    def set_floor_plane(self, floor_plane: FloorPlane):
        self._floor_plane = floor_plane
    
    def update(self, persons: Dict[int, PersonData], tracker: PersonTracker,
               now: float) -> Dict[int, bool]:
        """
        Update fall detection for all persons.
        Returns: Dict[person_id, is_new_alert]
        """
        alerts = {}
        current_pids = set(persons.keys())
        
        # Cleanup old states
        for pid in list(self.person_states.keys()):
            if pid not in current_pids:
                del self.person_states[pid]
        
        for pid, pdata in persons.items():
            if not tracker.has_entered_roi(pid):
                alerts[pid] = False
                continue
            
            # Initialize state
            if pid not in self.person_states:
                self.person_states[pid] = PersonFallState(person_id=pid)
            
            state = self.person_states[pid]
            
            # Update history
            state.center_history.append(pdata.center)
            if len(state.center_history) > 30:
                state.center_history = state.center_history[-30:]
            
            # Calibration phase
            if not state.is_calibrated:
                self._calibrate(state, pdata)
                alerts[pid] = False
                continue
            
            # Calculate all metrics
            self._calculate_metrics(state, pdata)
            
            # Evaluate active mode
            is_fall = self._evaluate_fall(state)
            
            if is_fall:
                state.consecutive_fall_frames += 1
            else:
                state.consecutive_fall_frames = max(0, state.consecutive_fall_frames - 2)
            
            # Confirm fall
            is_new_alert = False
            if (state.consecutive_fall_frames >= self.config.persist_frames and
                not state.is_fallen and
                now - state.last_alert_time > self.config.cooldown_sec):
                
                state.is_fallen = True
                state.fall_detected_at = now
                state.last_alert_time = now
                state.triggered_by = self.active_mode
                is_new_alert = True
                
                # Alarm in separate thread
                threading.Thread(target=beep_fall_alarm, daemon=True).start()
                
                print(f"[FALL] ⚠️ VAL GEDETECTEERD! Person {pid}")
                print(f"       Mode: {FallDetectionMode.get_name(self.active_mode)}")
                print(f"       Scores: {state.method_scores}")
                
                self._record_event(pid, pdata, state, now)
            
            alerts[pid] = is_new_alert
        
        return alerts
    
    def _calibrate(self, state: PersonFallState, pdata: PersonData):
        """Calibrate reference values when person is standing"""
        state.calibration_frames += 1
        
        x1, y1, x2, y2 = pdata.box
        height = y2 - y1
        
        # Need keypoints for good calibration
        if pdata.keypoints is None:
            return
        
        kp = pdata.keypoints
        
        # Check if person is roughly vertical (standing)
        if kp.shoulder_mid and kp.hip_mid:
            dx = abs(kp.shoulder_mid[0] - kp.hip_mid[0])
            dy = abs(kp.shoulder_mid[1] - kp.hip_mid[1])
            if dy > 0 and dx / dy > 0.5:  # Too horizontal
                return
        
        if state.calibration_frames >= self.config.calibration_frames:
            # Store reference values
            state.ref_height = height
            
            if kp.shoulder_mid:
                state.ref_torso_y = kp.shoulder_mid[1]
            
            if kp.hand_mid:
                state.ref_hand_y = kp.hand_mid[1]
            
            # 3D torso depth
            if self._depth_frame is not None and kp.shoulder_mid:
                depth = self._get_depth_at(kp.shoulder_mid[0], kp.shoulder_mid[1])
                if depth and depth > 0:
                    state.ref_torso_depth = depth
            
            state.is_calibrated = True
            print(f"[FALL] Person {state.person_id} calibrated: height={height:.0f}px")
    
    def _calculate_metrics(self, state: PersonFallState, pdata: PersonData):
        """Calculate all fall detection metrics"""
        cfg = self.config
        x1, y1, x2, y2 = pdata.box
        height = y2 - y1
        width = x2 - x1
        kp = pdata.keypoints
        
        # Reset scores
        state.method_scores = {}
        
        # === Get current depth ===
        current_depth = None
        if self._depth_frame is not None and kp and kp.shoulder_mid:
            current_depth = self._get_depth_at(kp.shoulder_mid[0], kp.shoulder_mid[1])
            if current_depth and current_depth > 0:
                state.current_depth = current_depth
        
        # === VELOCITY (simple 2D) ===
        if len(state.center_history) >= 3:
            recent = state.center_history[-5:]
            velocities = [recent[i+1][1] - recent[i][1] for i in range(len(recent)-1)]
            state.current_velocity = sum(velocities) / len(velocities)
            
            # Store for acceleration
            state.velocity_history.append(state.current_velocity)
            if len(state.velocity_history) > 10:
                state.velocity_history = state.velocity_history[-10:]
            
            # Velocity score
            score = min(1.0, max(0, state.current_velocity / cfg.velocity_thresh_px_per_frame))
            state.method_scores[FallDetectionMode.VELOCITY] = score
        
        # === ACCELERATION ===
        if len(state.velocity_history) >= 3:
            recent_vel = state.velocity_history[-5:]
            accelerations = [recent_vel[i+1] - recent_vel[i] for i in range(len(recent_vel)-1)]
            state.current_acceleration = sum(accelerations) / len(accelerations)
            
            accel_score = min(1.0, max(0, state.current_acceleration / cfg.acceleration_thresh))
            state.method_scores[FallDetectionMode.ACCELERATION] = accel_score
        
        # === TORSO_ANGLE ===
        if kp and kp.shoulder_mid and kp.hip_mid:
            dx = kp.shoulder_mid[0] - kp.hip_mid[0]
            dy = kp.shoulder_mid[1] - kp.hip_mid[1]
            # Angle from vertical (0 = straight up, 90 = horizontal)
            angle = math.degrees(math.atan2(abs(dx), abs(dy))) if dy != 0 else 90
            state.current_torso_angle = angle
            
            score = min(1.0, max(0, (angle - 30) / (cfg.torso_angle_thresh_deg - 30)))
            state.method_scores[FallDetectionMode.TORSO_ANGLE] = score
        
        # === LEG_ANGLE (knee to ankle) ===
        # Check both legs, use the most horizontal one
        leg_angles = []
        if kp:
            # Left leg: knee to ankle
            if kp.left_knee and kp.left_ankle:
                dx = kp.left_ankle[0] - kp.left_knee[0]
                dy = kp.left_ankle[1] - kp.left_knee[1]
                # Angle from horizontal (0 = horizontal, 90 = vertical)
                angle = math.degrees(math.atan2(abs(dy), abs(dx))) if dx != 0 else 90
                leg_angles.append(angle)
            
            # Right leg: knee to ankle
            if kp.right_knee and kp.right_ankle:
                dx = kp.right_ankle[0] - kp.right_knee[0]
                dy = kp.right_ankle[1] - kp.right_knee[1]
                angle = math.degrees(math.atan2(abs(dy), abs(dx))) if dx != 0 else 90
                leg_angles.append(angle)
        
        if leg_angles:
            # Use minimum angle (most horizontal leg)
            state.current_leg_angle = min(leg_angles)
            
            # Score: lower angle = more horizontal = more likely fallen
            # < 35° from horizontal = high score
            if state.current_leg_angle < cfg.leg_angle_thresh_deg:
                score = 1.0 - (state.current_leg_angle / cfg.leg_angle_thresh_deg)
            else:
                score = 0.0
            state.method_scores[FallDetectionMode.LEG_ANGLE] = score
        
        # === TORSO_FLOOR (3D) ===
        if (current_depth and self._floor_plane and self._floor_plane.is_defined and 
            kp and kp.shoulder_mid):
            x3d, y3d, z3d = pixel_to_3d(kp.shoulder_mid[0], kp.shoulder_mid[1], 
                                        current_depth, APP_CONFIG)
            dist = self._floor_plane.distance_to_point(x3d, y3d, z3d)
            state.current_torso_floor_dist = dist
            
            # Score based on both absolute and relative thresholds
            score1 = 1.0 if dist < cfg.torso_floor_thresh_mm else 0.0
            score2 = 0.0
            if state.ref_torso_depth:
                ratio = current_depth / state.ref_torso_depth
                score2 = 1.0 if ratio < cfg.torso_floor_ratio else 0.0
            
            state.method_scores[FallDetectionMode.TORSO_FLOOR] = max(score1, score2)
        
        # === HAND_POSITION ===
        if kp and kp.hand_mid and state.ref_hand_y and state.ref_height:
            hand_drop = kp.hand_mid[1] - state.ref_hand_y
            drop_ratio = hand_drop / state.ref_height
            state.current_hand_drop = drop_ratio
            
            score = min(1.0, max(0, drop_ratio / cfg.hand_drop_ratio))
            state.method_scores[FallDetectionMode.HAND_POSITION] = score
        
        # === HAND_FEET_OVERLAP ===
        if kp and kp.hand_mid and kp.feet_mid and state.ref_height:
            dist = abs(kp.hand_mid[1] - kp.feet_mid[1])
            dist_ratio = dist / state.ref_height
            state.current_hand_feet_dist = dist_ratio
            
            score = 1.0 - min(1.0, dist_ratio / cfg.hand_feet_dist_ratio)
            state.method_scores[FallDetectionMode.HAND_FEET_OVERLAP] = max(0, score)
        
        # === BODY_RATIO ===
        if state.ref_height and state.ref_height > 0:
            aspect_ratio = width / height if height > 0 else 1.0
            height_ratio = height / state.ref_height
            state.current_aspect_ratio = aspect_ratio
            state.current_height_ratio = height_ratio
            
            ar_score = min(1.0, max(0, (aspect_ratio - 0.5) / (cfg.aspect_ratio_thresh - 0.5)))
            hr_score = 1.0 - min(1.0, height_ratio / cfg.height_collapse_ratio)
            
            state.method_scores[FallDetectionMode.BODY_RATIO] = max(ar_score, max(0, hr_score))
    
    def _evaluate_fall(self, state: PersonFallState) -> bool:
        """Evaluate if fall is detected based on active mode"""
        scores = state.method_scores
        cfg = self.config
        
        if self.active_mode == FallDetectionMode.SIMPLE_OR:
            # Simple OR: velocity OR torso_angle OR leg_angle
            # Any one of these triggering = fall detected
            vel_triggered = scores.get(FallDetectionMode.VELOCITY, 0) > 0.7
            torso_triggered = scores.get(FallDetectionMode.TORSO_ANGLE, 0) > 0.7
            leg_triggered = scores.get(FallDetectionMode.LEG_ANGLE, 0) > 0.7
            
            return vel_triggered or torso_triggered or leg_triggered
        
        elif self.active_mode == FallDetectionMode.COMBINED:
            # Weighted combination
            total = 0.0
            weights_used = 0.0
            
            if FallDetectionMode.VELOCITY in scores:
                total += cfg.weight_velocity * scores[FallDetectionMode.VELOCITY]
                weights_used += cfg.weight_velocity
            
            if FallDetectionMode.ACCELERATION in scores:
                total += cfg.weight_acceleration * scores[FallDetectionMode.ACCELERATION]
                weights_used += cfg.weight_acceleration
            
            if FallDetectionMode.TORSO_ANGLE in scores:
                total += cfg.weight_torso_angle * scores[FallDetectionMode.TORSO_ANGLE]
                weights_used += cfg.weight_torso_angle
            
            if FallDetectionMode.TORSO_FLOOR in scores:
                total += cfg.weight_torso_floor * scores[FallDetectionMode.TORSO_FLOOR]
                weights_used += cfg.weight_torso_floor
            
            if FallDetectionMode.HAND_POSITION in scores:
                total += cfg.weight_hand_position * scores[FallDetectionMode.HAND_POSITION]
                weights_used += cfg.weight_hand_position
            
            if FallDetectionMode.HAND_FEET_OVERLAP in scores:
                total += cfg.weight_hand_feet * scores[FallDetectionMode.HAND_FEET_OVERLAP]
                weights_used += cfg.weight_hand_feet
            
            if FallDetectionMode.BODY_RATIO in scores:
                total += cfg.weight_body_ratio * scores[FallDetectionMode.BODY_RATIO]
                weights_used += cfg.weight_body_ratio
            
            if weights_used > 0:
                total /= weights_used
            
            # Combined needs at least 2 methods with high scores
            high_scores = sum(1 for s in scores.values() if s > 0.6)
            return total > 0.5 and high_scores >= 2
        
        elif self.active_mode == FallDetectionMode.VELOCITY_AND_ANGLE:
            # Both velocity AND angle must trigger (strictest mode)
            vel_score = scores.get(FallDetectionMode.VELOCITY, 0)
            angle_score = scores.get(FallDetectionMode.TORSO_ANGLE, 0)
            return vel_score > 0.6 and angle_score > 0.6
        
        else:
            # Single mode - check if that specific method triggers
            if self.active_mode in scores:
                return scores[self.active_mode] > 0.7
            return False
    
    def _get_depth_at(self, x: int, y: int) -> Optional[float]:
        """Get depth value at pixel location"""
        if self._depth_frame is None:
            return None
        
        dh, dw = self._depth_frame.shape
        dx = int(x * dw / APP_CONFIG.color_width)
        dy = int(y * dh / APP_CONFIG.color_height)
        
        if 0 <= dx < dw and 0 <= dy < dh:
            # Sample 3x3 area
            x_start, x_end = max(0, dx-1), min(dw, dx+2)
            y_start, y_end = max(0, dy-1), min(dh, dy+2)
            region = self._depth_frame[y_start:y_end, x_start:x_end]
            valid = region[region > 0]
            if len(valid) > 0:
                return float(np.median(valid))
        return None
    
    def _record_event(self, pid: int, pdata: PersonData, state: PersonFallState, now: float):
        """Record fall event"""
        event = {
            'timestamp': now,
            'person_id': pid,
            'mode': self.active_mode.name,
            'scores': state.method_scores.copy(),
            'velocity': state.current_velocity,
            'torso_angle': state.current_torso_angle,
            'torso_floor_dist': state.current_torso_floor_dist,
            'box': pdata.box,
            'center': pdata.center
        }
        self._fall_events.append(event)
    
    def get_person_state(self, pid: int) -> Optional[PersonFallState]:
        return self.person_states.get(pid)
    
    def get_pending_events(self) -> List[Dict]:
        events = self._fall_events.copy()
        self._fall_events.clear()
        return events


# ===================== FALL LOGGER =====================
class FallLogger:
    """Logs fall events to CSV"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        
        if not os.path.isfile(filepath):
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'person_id', 'detection_mode',
                    'velocity', 'torso_angle', 'torso_floor_dist',
                    'score_velocity', 'score_angle', 'score_floor',
                    'score_hand_pos', 'score_hand_feet', 'score_ratio',
                    'box_x1', 'box_y1', 'box_x2', 'box_y2'
                ])
            print(f"[FALL] Created log: {filepath}")
    
    def log_events(self, events: List[Dict]):
        if not events:
            return
        
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            for e in events:
                scores = e.get('scores', {})
                box = e.get('box', (0,0,0,0))
                writer.writerow([
                    datetime.fromtimestamp(e['timestamp']).strftime("%Y-%m-%d %H:%M:%S"),
                    e['person_id'],
                    e.get('mode', ''),
                    f"{e.get('velocity', 0):.2f}",
                    f"{e.get('torso_angle', 0):.1f}",
                    f"{e.get('torso_floor_dist', 0):.0f}",
                    f"{scores.get(FallDetectionMode.VELOCITY, 0):.2f}",
                    f"{scores.get(FallDetectionMode.TORSO_ANGLE, 0):.2f}",
                    f"{scores.get(FallDetectionMode.TORSO_FLOOR, 0):.2f}",
                    f"{scores.get(FallDetectionMode.HAND_POSITION, 0):.2f}",
                    f"{scores.get(FallDetectionMode.HAND_FEET_OVERLAP, 0):.2f}",
                    f"{scores.get(FallDetectionMode.BODY_RATIO, 0):.2f}",
                    box[0], box[1], box[2], box[3]
                ])
        
        for e in events:
            print(f"[FALL] Logged: Person {e['person_id']} - {e.get('mode', '')}")


# ===================== DETECTION =====================
def extract_keypoints(kp_array: np.ndarray) -> KeypointData:
    """Extract all keypoints from YOLO pose output"""
    
    def get_kp(idx: int) -> Optional[Tuple[int, int]]:
        if idx < len(kp_array):
            x, y = kp_array[idx][:2]
            if x > 0 and y > 0:
                return (int(x), int(y))
        return None
    
    return KeypointData(
        raw=kp_array,
        nose=get_kp(KeypointIdx.NOSE),
        left_shoulder=get_kp(KeypointIdx.LEFT_SHOULDER),
        right_shoulder=get_kp(KeypointIdx.RIGHT_SHOULDER),
        left_elbow=get_kp(KeypointIdx.LEFT_ELBOW),
        right_elbow=get_kp(KeypointIdx.RIGHT_ELBOW),
        left_wrist=get_kp(KeypointIdx.LEFT_WRIST),
        right_wrist=get_kp(KeypointIdx.RIGHT_WRIST),
        left_hip=get_kp(KeypointIdx.LEFT_HIP),
        right_hip=get_kp(KeypointIdx.RIGHT_HIP),
        left_knee=get_kp(KeypointIdx.LEFT_KNEE),
        right_knee=get_kp(KeypointIdx.RIGHT_KNEE),
        left_ankle=get_kp(KeypointIdx.LEFT_ANKLE),
        right_ankle=get_kp(KeypointIdx.RIGHT_ANKLE)
    )


def extract_persons_from_result(result, conf_thresh: float = 0.5) -> List[PersonData]:
    """Extract person detections with full keypoints"""
    detections = []
    
    if not hasattr(result, "boxes") or result.boxes is None:
        return detections
    
    boxes = result.boxes
    xyxy = boxes.xyxy
    cls = boxes.cls
    conf = boxes.conf
    
    if xyxy is None or cls is None or conf is None:
        return detections
    
    if hasattr(xyxy, "cpu"):
        xyxy = xyxy.cpu().numpy()
        cls = cls.cpu().numpy()
        conf = conf.cpu().numpy()
    
    keypoints = None
    if hasattr(result, "keypoints") and result.keypoints is not None:
        kp = result.keypoints
        if hasattr(kp, "xy"):
            keypoints = kp.xy
            if hasattr(keypoints, "cpu"):
                keypoints = keypoints.cpu().numpy()
    
    for i in range(len(xyxy)):
        if conf[i] < conf_thresh or int(cls[i]) != 0:
            continue
        
        x1, y1, x2, y2 = xyxy[i]
        cx = int(0.5 * (x1 + x2))
        cy = int(0.5 * (y1 + y2))
        
        kp_data = None
        feet = []
        
        if keypoints is not None and i < len(keypoints):
            kp_array = keypoints[i]
            kp_data = extract_keypoints(kp_array)
            
            # Extract feet
            if kp_data.left_ankle:
                feet.append(kp_data.left_ankle)
            if kp_data.right_ankle:
                feet.append(kp_data.right_ankle)
        
        if not feet:
            feet = [(cx, int(y2))]
        
        detections.append(PersonData(
            center=(cx, cy),
            box=(int(x1), int(y1), int(x2), int(y2)),
            feet=feet,
            keypoints=kp_data
        ))
    
    return detections


# ===================== CAMERA =====================
def init_camera():
    """Initialize Orbbec camera with color and depth"""
    pipe = Pipeline()
    cfg = Config()
    
    # Color stream
    color_prof = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR) \
        .get_video_stream_profile(APP_CONFIG.color_width, APP_CONFIG.color_height, 
                                  OBFormat.RGB, 30)
    cfg.enable_stream(color_prof)
    
    # Depth stream
    try:
        depth_prof = pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR) \
            .get_video_stream_profile(APP_CONFIG.depth_width, APP_CONFIG.depth_height,
                                      OBFormat.Y16, 30)
        cfg.enable_stream(depth_prof)
        print("[CAMERA] Depth stream enabled")
    except Exception as e:
        print(f"[CAMERA] Depth stream not available: {e}")
    
    pipe.start(cfg)
    return pipe


def get_frames(pipe, timeout_ms: int):
    """Get color and depth frames"""
    fs = pipe.wait_for_frames(timeout_ms)
    if fs is None:
        return None, None
    
    fs = fs.as_frame_set()
    
    # Color
    color_bgr = None
    cf = fs.get_color_frame()
    if cf is not None:
        vf = cf.as_video_frame()
        w, h = vf.get_width(), vf.get_height()
        arr = np.frombuffer(vf.get_data(), np.uint8).reshape(h, w, 3)
        color_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    
    # Depth
    depth_data = None
    df = fs.get_depth_frame()
    if df is not None:
        vf = df.as_video_frame()
        w, h = vf.get_width(), vf.get_height()
        depth_data = np.frombuffer(vf.get_data(), np.uint16).reshape(h, w)
    
    return color_bgr, depth_data


# ===================== VISUALIZATION =====================
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0),
]

def get_color(pid: int):
    return COLORS[pid % len(COLORS)]


def draw_keypoints(img, kp: KeypointData, color):
    """Draw skeleton keypoints"""
    points = [
        kp.nose, kp.left_shoulder, kp.right_shoulder,
        kp.left_elbow, kp.right_elbow, kp.left_wrist, kp.right_wrist,
        kp.left_hip, kp.right_hip, kp.left_knee, kp.right_knee,
        kp.left_ankle, kp.right_ankle
    ]
    
    # Draw points
    for p in points:
        if p:
            cv2.circle(img, p, 4, color, -1)
    
    # Draw connections
    connections = [
        (kp.left_shoulder, kp.right_shoulder),
        (kp.left_shoulder, kp.left_elbow),
        (kp.left_elbow, kp.left_wrist),
        (kp.right_shoulder, kp.right_elbow),
        (kp.right_elbow, kp.right_wrist),
        (kp.left_shoulder, kp.left_hip),
        (kp.right_shoulder, kp.right_hip),
        (kp.left_hip, kp.right_hip),
        (kp.left_hip, kp.left_knee),
        (kp.left_knee, kp.left_ankle),
        (kp.right_hip, kp.right_knee),
        (kp.right_knee, kp.right_ankle),
    ]
    
    for p1, p2 in connections:
        if p1 and p2:
            cv2.line(img, p1, p2, color, 2)


def draw_floor_plane(img, floor_plane: FloorPlane):
    """Draw floor plane points and polygon"""
    if not floor_plane.points_2d:
        return
    
    # Draw points
    for i, p in enumerate(floor_plane.points_2d):
        cv2.circle(img, p, 8, (0, 255, 255), -1)
        cv2.putText(img, f"F{i+1}", (p[0]+10, p[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Draw polygon if enough points
    if len(floor_plane.points_2d) >= 3:
        pts = np.array(floor_plane.points_2d, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 255, 255), 2)
        
        # Semi-transparent fill
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 255))
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)


def draw_setup_mode(img, zone_manager: ZoneManager, floor_plane: FloorPlane):
    """Draw setup mode overlay"""
    overlay = img.copy()
    
    # Draw floor plane
    draw_floor_plane(overlay, floor_plane)
    
    # Draw preview
    preview = zone_manager.draw_preview
    if preview:
        ptype, rect = preview
        color = (255, 0, 255) if ptype == "roi" else (255, 255, 0)
        label = "Drawing ROI..." if ptype == "roi" else f"Drawing Zone {zone_manager.current_segment}..."
        cv2.rectangle(overlay, (rect[0], rect[1]), (rect[2], rect[3]), color, 3)
        cv2.putText(overlay, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Draw ROI
    if zone_manager.stair_roi:
        x1, y1, x2, y2 = zone_manager.stair_roi
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(overlay, "STAIR ROI", (x1+5, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    # Draw segment zones
    for i, zone in enumerate(zone_manager.segment_zones):
        if zone is None:
            continue
        x1, y1, x2, y2 = zone
        is_current = (i == zone_manager.current_segment)
        color = (0, 255, 255) if is_current else (0, 255, 0)
        thick = 3 if is_current else 2
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thick)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(overlay, f"Z{i}", (cx-20, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Instructions
    y = 70
    cv2.putText(overlay, f"Zone: {zone_manager.current_segment} (0-3 to change)", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    y += 30
    cv2.putText(overlay, "LEFT DRAG = Draw zone | RIGHT DRAG = Draw ROI", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y += 25
    cv2.putText(overlay, "MIDDLE CLICK = Add floor point | F = Reset floor", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y += 30
    
    floor_ok = floor_plane.is_defined
    floor_text = f"Floor: {'DEFINED' if floor_ok else f'{len(floor_plane.points_2d)}/3 points'}"
    floor_color = (0, 255, 0) if floor_ok else (0, 0, 255)
    cv2.putText(overlay, floor_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, floor_color, 2)
    y += 30
    
    roi_ok = zone_manager.stair_roi is not None
    roi_text = f"ROI: {'DEFINED' if roi_ok else 'NOT SET'}"
    roi_color = (0, 255, 0) if roi_ok else (0, 0, 255)
    cv2.putText(overlay, roi_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_color, 2)
    y += 30
    
    if zone_manager.all_defined():
        cv2.putText(overlay, "Press 'C' to START TRACKING", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return overlay


def draw_tracking_mode(img, zone_manager: ZoneManager, tracker: PersonTracker,
                       fall_detector: FallDetector, floor_plane: FloorPlane,
                       fall_alerts: Dict[int, bool], active_touches: ActiveTouchState,
                       touch_assigner: TouchAssigner, logger: RailingUsageLogger):
    """Draw tracking mode overlay with feedback messages"""
    overlay = img.copy()
    now = time.time()
    
    # Draw floor plane
    draw_floor_plane(overlay, floor_plane)
    
    # Draw ROI
    if zone_manager.stair_roi:
        x1, y1, x2, y2 = zone_manager.stair_roi
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), 2)
    
    # Draw segment zones (with touch highlight)
    for i, zone in enumerate(zone_manager.segment_zones):
        if zone is None:
            continue
        x1, y1, x2, y2 = zone
        
        is_touched = active_touches.is_touched(i)
        
        if is_touched:
            # Highlight touched zone
            color = (0, 255, 255)
            thick = 5
            # Semi-transparent overlay
            sub_img = overlay[y1:y2, x1:x2]
            if sub_img.size > 0:
                highlight = np.ones(sub_img.shape, dtype=np.uint8) * np.array([0, 255, 255], dtype=np.uint8)
                overlay[y1:y2, x1:x2] = cv2.addWeighted(sub_img, 0.6, highlight, 0.4, 0)
        else:
            color = (100, 100, 100)
            thick = 2
        
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thick)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        label = f"Z{i}" + (" TOUCHED" if is_touched else "")
        label_color = (0, 255, 255) if is_touched else (150, 150, 150)
        cv2.putText(overlay, label, (cx-30, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
    
    # Draw persons
    for pid, pdata in tracker.persons.items():
        in_roi = tracker.has_entered_roi(pid)
        fall_state = fall_detector.get_person_state(pid)
        is_fallen = fall_state.is_fallen if fall_state else False
        
        # Color based on state
        if is_fallen:
            color = (0, 0, 255)
            thick = 4
        elif in_roi:
            color = get_color(pid)
            thick = 3
        else:
            color = (100, 100, 100)
            thick = 1
        
        x1, y1, x2, y2 = pdata.box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thick)
        
        # Label
        label = f"P{pid}"
        if is_fallen:
            label += " [GEVALLEN]"
        elif not in_roi:
            label += " (waiting)"
        cv2.putText(overlay, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw skeleton
        if pdata.keypoints and in_roi:
            draw_keypoints(overlay, pdata.keypoints, color)
        
        # Draw feet
        if in_roi:
            for foot in pdata.feet:
                cv2.circle(overlay, foot, 6, color, -1)
        
        # Fall alert - takes priority
        if is_fallen:
            fall_y = max(30, y1 - 100)
            fall_msg = "⚠️ VAL GEDETECTEERD! ⚠️"
            
            flash = int(now * 4) % 2 == 0
            bg_color = (0, 0, 255) if flash else (0, 0, 180)
            
            (tw, th), _ = cv2.getTextSize(fall_msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            cv2.rectangle(overlay, (x1-10, fall_y-th-15), (x1+tw+10, fall_y+10), bg_color, -1)
            cv2.rectangle(overlay, (x1-10, fall_y-th-15), (x1+tw+10, fall_y+10), (255, 255, 255), 3)
            cv2.putText(overlay, fall_msg, (x1, fall_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            continue  # Skip normal feedback when fallen
        
        # === FEEDBACK MESSAGES ABOVE HEAD ===
        if in_roi:
            segments = touch_assigner.get_person_segments(pid)
            num_segments = len(segments)
            
            # Show touched segments below box
            if segments:
                seg_str = ",".join(f"Z{s}" for s in sorted(segments))
                cv2.putText(overlay, f"Touched: {seg_str}", (x1, y2+25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Feedback message based on segments touched - ABOVE HEAD
            feedback_y = y1 - 40
            entry_time = logger.get_entry_time(pid)
            time_in_roi = (now - entry_time) if entry_time else 0
            
            if num_segments >= 2:
                # SAFE - good job!
                feedback_msg = "Je bent veilig bezig topper!!"
                feedback_color = (0, 255, 0)  # Green
                show_fact = False
                show_positive_msg = time_in_roi > 3.0
            elif num_segments == 1:
                # Partial - could be better
                feedback_msg = "Je hebt maar 1 keer de leuning vastgehouden"
                feedback_color = (0, 165, 255)  # Orange
                show_fact = False
                show_positive_msg = False
            else:
                # UNSAFE - no railing
                feedback_msg = "Gelieve de leuning vast te houden in het vervolg"
                feedback_color = (0, 0, 255)  # Red
                show_fact = time_in_roi > 5.0
                show_positive_msg = False
            
            cv2.putText(overlay, feedback_msg, (x1, feedback_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, feedback_color, 2)
            
            # Show positive stats message after 3+ seconds with good railing use
            if show_positive_msg:
                safe_count = logger.get_safe_users_today() + 1
                positive_msg = f"Goed zo! Je bent 1 van de {safe_count} veilige gebruikers vandaag!"
                
                positive_y = y1 - 70
                (text_w, text_h), _ = cv2.getTextSize(positive_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(overlay, (x1-5, positive_y-text_h-5), 
                             (x1+text_w+5, positive_y+5), (0, 100, 0), -1)
                cv2.putText(overlay, positive_msg, (x1, positive_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show safety fact after 5+ seconds without railing
            elif show_fact:
                fact_idx = pid % len(SAFETY_FACTS)
                fact = SAFETY_FACTS[fact_idx]
                
                fact_y = y1 - 70
                (text_w, text_h), _ = cv2.getTextSize(fact, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(overlay, (x1-5, fact_y-text_h-5), 
                             (x1+text_w+5, fact_y+5), (0, 0, 100), -1)
                cv2.putText(overlay, fact, (x1, fact_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show fall detection metrics below
            if fall_state and fall_state.is_calibrated:
                y_offset = y2 + 45
                scores = fall_state.method_scores
                
                # Show active mode score
                active_mode = fall_detector.active_mode
                if active_mode == FallDetectionMode.SIMPLE_OR:
                    # Show all 3 scores for OR mode
                    vel = scores.get(FallDetectionMode.VELOCITY, 0)
                    torso = scores.get(FallDetectionMode.TORSO_ANGLE, 0)
                    leg = scores.get(FallDetectionMode.LEG_ANGLE, 0)
                    
                    any_triggered = vel > 0.7 or torso > 0.7 or leg > 0.7
                    score_color = (0, 0, 255) if any_triggered else (0, 255, 0)
                    
                    cv2.putText(overlay, f"V:{vel:.1f} T:{torso:.1f} L:{leg:.1f}", (x1, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, score_color, 1)
                    y_offset += 15
                elif active_mode == FallDetectionMode.VELOCITY_AND_ANGLE:
                    vel_score = scores.get(FallDetectionMode.VELOCITY, 0)
                    angle_score = scores.get(FallDetectionMode.TORSO_ANGLE, 0)
                    score_color = (0, 255, 0) if (vel_score < 0.6 and angle_score < 0.6) else (0, 0, 255)
                    cv2.putText(overlay, f"V:{vel_score:.2f} A:{angle_score:.2f}", (x1, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, score_color, 1)
                    y_offset += 15
                elif active_mode in scores:
                    score = scores[active_mode]
                    score_color = (0, 255, 0) if score < 0.5 else (0, 165, 255) if score < 0.7 else (0, 0, 255)
                    cv2.putText(overlay, f"{active_mode.name}: {score:.2f}", (x1, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, score_color, 1)
                    y_offset += 15
                
                # Show key metrics
                cv2.putText(overlay, f"Vel:{fall_state.current_velocity:.1f} Ang:{fall_state.current_torso_angle:.0f} Leg:{fall_state.current_leg_angle:.0f}", 
                            (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    
    # Stats panel
    cv2.putText(overlay, f"Persons: {len(tracker.persons)}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    mode_name = FallDetectionMode.get_name(fall_detector.active_mode)
    cv2.putText(overlay, f"Mode: {mode_name}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Active touches
    active = active_touches.active_segments
    if active:
        zones_str = ", ".join(f"Z{z}" for z in sorted(active))
        cv2.putText(overlay, f"Active: {zones_str}", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    cv2.putText(overlay, "F4=SIMPLE_OR | M=Cycle | R=Setup | ESC=Quit", (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    return overlay


# ===================== MAIN =====================
def main():
    print("=" * 60)
    print("STAIR SAFETY TRACKER v4 - Multi-Mode Fall Detection")
    print("=" * 60)
    
    # Initialize
    print("[INIT] Loading YOLO model...")
    model = YOLO(APP_CONFIG.pose_checkpoint)
    try:
        model.to(DEVICE)
    except Exception:
        pass
    
    print("[INIT] Starting camera...")
    pipe = init_camera()
    
    print(f"[INIT] Starting UDP on port {APP_CONFIG.udp_port}...")
    udp = UDPReceiver(APP_CONFIG.udp_port, APP_CONFIG.num_segments)
    
    # Components
    zone_manager = ZoneManager(APP_CONFIG.num_segments)
    tracker = PersonTracker(APP_CONFIG.max_disappeared_frames)
    fall_detector = FallDetector(FALL_CONFIG)
    fall_logger = FallLogger(APP_CONFIG.fall_csv_output)
    floor_plane = FloorPlane()
    
    # Touch tracking components
    active_touches = ActiveTouchState()
    touch_assigner = TouchAssigner(zone_manager, APP_CONFIG.max_assign_distance_px)
    railing_logger = RailingUsageLogger(APP_CONFIG.csv_output)
    
    # Window
    window_name = "StairTracker v4 - Fall Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 800)
    
    # Mouse callback with depth access
    current_depth = [None]  # Mutable container for closure
    
    def mouse_callback(event, x, y, flags, param):
        zone_manager.mouse_callback(event, x, y, flags, param)
        
        # Middle click to add floor point
        if event == cv2.EVENT_MBUTTONDOWN:
            if current_depth[0] is not None and len(floor_plane.points_2d) < 4:
                dh, dw = current_depth[0].shape
                dx = int(x * dw / APP_CONFIG.color_width)
                dy = int(y * dh / APP_CONFIG.color_height)
                
                if 0 <= dx < dw and 0 <= dy < dh:
                    depth_val = current_depth[0][dy, dx]
                    if depth_val > 0:
                        floor_plane.add_point(x, y, float(depth_val), APP_CONFIG)
                    else:
                        print("[FLOOR] Invalid depth at this point")
    
    cv2.setMouseCallback(window_name, mouse_callback)
    
    mode = "setup"
    
    # Start with SIMPLE_OR mode (recommended)
    fall_detector.set_mode(FallDetectionMode.SIMPLE_OR)
    
    print()
    print("SETUP MODE:")
    print("  - Press 0-3 to select zone")
    print("  - LEFT DRAG = Draw segment zone")
    print("  - RIGHT DRAG = Draw stair ROI")
    print("  - MIDDLE CLICK = Add floor plane point (need 3)")
    print("  - Press 'F' to reset floor")
    print("  - Press 'C' when ready")
    print()
    print("TRACKING MODE:")
    print("  - F1 = VELOCITY (snelheid > 8px/f)")
    print("  - F2 = TORSO_ANGLE (hoek > 55°)")
    print("  - F3 = LEG_ANGLE (been horizontaal)")
    print("  - F4 = SIMPLE_OR ★ AANBEVOLEN (vel OF torso OF been)")
    print("  - F5 = ACCELERATION")
    print("  - F6 = VELOCITY+ANGLE (beide)")
    print("  - F7 = TORSO_FLOOR (3D)")
    print("  - M = Cycle modes")
    print("  - R = Return to setup")
    print("=" * 60)
    
    try:
        while True:
            frame, depth_frame = get_frames(pipe, APP_CONFIG.camera_timeout_ms)
            if frame is None:
                continue
            
            current_depth[0] = depth_frame
            
            if mode == "tracking":
                # Detection
                result = model.track(
                    source=frame[:, :, ::-1],
                    device=DEVICE,
                    verbose=False,
                    conf=APP_CONFIG.confidence,
                    imgsz=APP_CONFIG.image_size,
                    persist=True,
                    tracker="bytetrack.yaml"
                )[0]
                
                detections = extract_persons_from_result(result, APP_CONFIG.confidence)
                tracker.update(detections)
                tracker.check_roi_entry(zone_manager)
                
                # Get current time
                now = time.time()
                
                # Track ROI entry times for railing logger
                for pid in tracker.persons:
                    if tracker.has_entered_roi(pid):
                        railing_logger.track_roi_entry(pid, now)
                
                # Process UDP touch events
                for seg_id, event_type, ts in udp.poll():
                    active_touches.process_event(seg_id, event_type, ts)
                    touch_assigner.assign(seg_id, event_type, tracker, ts)
                
                # Try to assign unassigned touches
                touch_assigner.try_assign_unassigned(tracker)
                
                # Log persons who left ROI
                for pid in tracker.get_left_roi_persons():
                    railing_logger.log_person(pid, touch_assigner, now)
                
                # Fall detection
                fall_detector.set_depth_frame(depth_frame)
                fall_detector.set_floor_plane(floor_plane)
                fall_alerts = fall_detector.update(tracker.persons, tracker, now)
                
                # Log falls
                events = fall_detector.get_pending_events()
                if events:
                    fall_logger.log_events(events)
                
                # Draw
                display = draw_tracking_mode(frame, zone_manager, tracker, 
                                            fall_detector, floor_plane, fall_alerts,
                                            active_touches, touch_assigner, railing_logger)
            else:
                display = draw_setup_mode(frame, zone_manager, floor_plane)
            
            cv2.imshow(window_name, display)
            
            # Keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('r') or key == ord('R'):
                if mode == "tracking":
                    mode = "setup"
                    tracker = PersonTracker(APP_CONFIG.max_disappeared_frames)
                    fall_detector = FallDetector(FALL_CONFIG)
                    active_touches = ActiveTouchState()
                    touch_assigner = TouchAssigner(zone_manager, APP_CONFIG.max_assign_distance_px)
                    railing_logger = RailingUsageLogger(APP_CONFIG.csv_output)
                    print("[MODE] Returned to SETUP")
                else:
                    zone_manager.reset()
                    print("[SETUP] Zones reset")
            elif key == ord('f') or key == ord('F'):
                floor_plane.reset()
                print("[FLOOR] Floor plane reset")
            elif key == ord('c') or key == ord('C'):
                if mode == "setup" and zone_manager.all_defined():
                    mode = "tracking"
                    print("[MODE] TRACKING started")
                elif mode == "setup":
                    print("[MODE] Define all zones and ROI first!")
            elif key == ord('m') or key == ord('M'):
                fall_detector.cycle_mode()
            elif key in [ord('0'), ord('1'), ord('2'), ord('3')]:
                zone_manager.current_segment = key - ord('0')
                print(f"[ZONE] Selected zone {zone_manager.current_segment}")
            
            # F-keys for detection modes
            elif key == 190:  # F1
                fall_detector.set_mode(FallDetectionMode.VELOCITY)
            elif key == 191:  # F2
                fall_detector.set_mode(FallDetectionMode.TORSO_ANGLE)
            elif key == 192:  # F3
                fall_detector.set_mode(FallDetectionMode.LEG_ANGLE)
            elif key == 193:  # F4 - RECOMMENDED
                fall_detector.set_mode(FallDetectionMode.SIMPLE_OR)
            elif key == 194:  # F5
                fall_detector.set_mode(FallDetectionMode.ACCELERATION)
            elif key == 195:  # F6
                fall_detector.set_mode(FallDetectionMode.VELOCITY_AND_ANGLE)
            elif key == 196:  # F7
                fall_detector.set_mode(FallDetectionMode.TORSO_FLOOR)
            elif key == 197:  # F8
                fall_detector.set_mode(FallDetectionMode.HAND_POSITION)
            elif key == 198:  # F9
                fall_detector.set_mode(FallDetectionMode.COMBINED)
    
    finally:
        try:
            pipe.stop()
        except:
            pass
        udp.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
