import cv2
import time
import pygame
import os
import numpy as np
from scipy.spatial import distance as dist
from collections import deque

# Initialize pygame mixer for audio
pygame.mixer.init()

# Load alarm sound
alarm_path = os.path.join("drowsiness_model", "alarm.wav")
if os.path.exists(alarm_path):
    alarm_sound = pygame.mixer.Sound(alarm_path)
else:
    print(f"Warning: {alarm_path} not found. Alarm won't work.")
    alarm_sound = None

# Initialize dlib's face detector and facial landmark predictor
import dlib

detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"

if not os.path.exists(predictor_path):
    print("ERROR: 'shape_predictor_68_face_landmarks.dat' file nahi mili!")
    print("Download karein: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    print("Extract karke is project folder mein rakhen.")
    exit()

predictor = dlib.shape_predictor(predictor_path)

# Thresholds
EAR_CLOSED_THRESHOLD = 0.21
EAR_SLEEPY_THRESHOLD = 0.25
MAR_YAWN_THRESHOLD = 0.6

# Counters
eyes_closed_counter = 0
sleepy_eyes_counter = 0
total_blinks = 0
yawns_counter = 0
head_nods_counter = 0

# Frame counters
EYE_CLOSED_FRAMES = 0
SLEEPY_FRAMES = 0
BLINK_FRAMES_MIN = 2
BLINK_FRAMES_MAX = 8

# Blink detection states
was_blinking = False

# Head nodding detection - IMPROVED
HEAD_Y_POSITIONS = deque(maxlen=60)  # 2 seconds history
head_movement_state = "NEUTRAL"  # NEUTRAL, MOVING_DOWN, MOVING_UP
last_peak = None
last_valley = None
nod_cycle_started = False

# Yawn detection
YAWN_FRAMES = 0
YAWN_CONSECUTIVE_THRESHOLD = 15
last_yawn_time = 0
YAWN_COOLDOWN = 3  # seconds

alarm_playing = False

def reset_counters():
    """Reset all counters"""
    global eyes_closed_counter, sleepy_eyes_counter, total_blinks
    global yawns_counter, head_nods_counter
    eyes_closed_counter = 0
    sleepy_eyes_counter = 0
    total_blinks = 0
    yawns_counter = 0
    head_nods_counter = 0
    print("\n[RESET] All counters reset to 0!")

def eye_aspect_ratio(eye):
    """Calculate Eye Aspect Ratio (EAR)"""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    """Calculate Mouth Aspect Ratio (MAR)"""
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

def detect_head_nodding_simple():
    """Simple and effective head nodding detection"""
    global head_movement_state, last_peak, last_valley, nod_cycle_started
    
    if len(HEAD_Y_POSITIONS) < 20:
        return False
    
    positions = list(HEAD_Y_POSITIONS)
    
    # Get current position and recent average
    current_pos = positions[-1]
    recent_avg = np.mean(positions[-10:])
    older_avg = np.mean(positions[-20:-10])
    
    # Calculate movement direction
    movement = recent_avg - older_avg
    
    # Detect downward movement (head going down)
    if movement > 3 and head_movement_state != "MOVING_DOWN":
        head_movement_state = "MOVING_DOWN"
        last_peak = current_pos
        nod_cycle_started = True
    
    # Detect upward movement (head coming up)
    elif movement < -3 and head_movement_state == "MOVING_DOWN" and nod_cycle_started:
        head_movement_state = "MOVING_UP"
        last_valley = current_pos
        
        # Check if it's a valid nod (sufficient movement)
        if last_peak and last_valley:
            movement_range = abs(last_peak - last_valley)
            if movement_range > 8:  # Minimum 8 pixels movement
                nod_cycle_started = False
                head_movement_state = "NEUTRAL"
                return True
    
    # Reset if movement stops
    elif abs(movement) < 1:
        if head_movement_state == "MOVING_UP":
            head_movement_state = "NEUTRAL"
    
    return False

def shape_to_np(shape, dtype="int"):
    """Convert dlib shape to numpy array"""
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def play_alarm():
    global alarm_playing
    if alarm_sound and not alarm_playing:
        alarm_sound.play(-1)
        alarm_playing = True

def stop_alarm():
    global alarm_playing
    if alarm_sound and alarm_playing:
        alarm_sound.stop()
        alarm_playing = False

# Facial landmark indices
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))

cap = cv2.VideoCapture(0)
frame_counter = 0

print("Drowsiness Detection System Started!")
print("Press 'Q' to quit")
print("Press 'R' to reset counters")
print("\n=== THRESHOLDS ===")
print(f"EAR Closed: < {EAR_CLOSED_THRESHOLD}")
print(f"EAR Sleepy: < {EAR_SLEEPY_THRESHOLD}")
print(f"MAR Yawn: > {MAR_YAWN_THRESHOLD}")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_counter += 1
    current_time = time.time()
    
    h, w = frame.shape[:2]
    
    faces = detector(gray, 0)
    
    eye_status = "OPEN"
    ear_left = 0.0
    ear_right = 0.0
    mar_value = 0.0
    
    drowsiness_alert = False
    yawning_detected = False
    head_nodding_detected = False
    
    for face in faces:
        x, y, w_face, h_face = face.left(), face.top(), face.width(), face.height()
        
        # Get facial landmarks
        shape = predictor(gray, face)
        shape = shape_to_np(shape)
        
        leftEye = shape[LEFT_EYE]
        rightEye = shape[RIGHT_EYE]
        mouth = shape[MOUTH]
        
        # Calculate ratios
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear_value = (leftEAR + rightEAR) / 2.0
        mar_value = mouth_aspect_ratio(mouth)
        
        ear_left = leftEAR
        ear_right = rightEAR
        
        # Head position tracking - use chin position for better detection
        chin_y = int(shape[8][1])  # Bottom of chin
        nose_y = int(shape[30][1])  # Nose tip
        head_y = int((chin_y + nose_y) / 2)
        
        HEAD_Y_POSITIONS.append(head_y)
        
        # Detect head nodding
        head_nodding_detected = detect_head_nodding_simple()
        if head_nodding_detected:
            head_nods_counter += 1
            print(f"[NOD DETECTED] Total: {head_nods_counter}")
        
        # === IMPROVED BLINK & SLEEPY DETECTION ===
        if ear_value < EAR_CLOSED_THRESHOLD:
            # Eyes CLOSED
            EYE_CLOSED_FRAMES += 1
            eye_status = "CLOSED"
            eye_color = (0, 0, 255)  # RED
            
            # Reset sleepy counter when eyes close
            SLEEPY_FRAMES = 0
            
            # If closed too long, it's drowsiness
            if EYE_CLOSED_FRAMES > BLINK_FRAMES_MAX:
                drowsiness_alert = True
                if EYE_CLOSED_FRAMES == BLINK_FRAMES_MAX + 1:
                    eyes_closed_counter += 1
            
            was_blinking = True
            
        elif ear_value < EAR_SLEEPY_THRESHOLD:
            # Eyes SLEEPY (half-closed)
            eye_status = "SLEEPY"
            eye_color = (0, 255, 255)  # YELLOW
            
            # Count blink if transitioning from closed
            if was_blinking and BLINK_FRAMES_MIN <= EYE_CLOSED_FRAMES <= BLINK_FRAMES_MAX:
                total_blinks += 1
                print(f"[BLINK] Total: {total_blinks}")
                was_blinking = False
            
            # Increment sleepy counter
            SLEEPY_FRAMES += 1
            EYE_CLOSED_FRAMES = 0
            
            # Count as sleepy event every 45 frames (1.5 seconds)
            if SLEEPY_FRAMES > 0 and SLEEPY_FRAMES % 45 == 0:
                sleepy_eyes_counter += 1
                print(f"[SLEEPY] Total: {sleepy_eyes_counter}")
                
        else:
            # Eyes OPEN
            eye_status = "OPEN"
            eye_color = (0, 255, 0)  # GREEN
            
            # Count blink if transitioning from closed
            if was_blinking and BLINK_FRAMES_MIN <= EYE_CLOSED_FRAMES <= BLINK_FRAMES_MAX:
                total_blinks += 1
                print(f"[BLINK] Total: {total_blinks}")
                was_blinking = False
            
            # If eyes were sleepy, count it
            if SLEEPY_FRAMES >= 30:  # At least 1 second of sleepy state
                if SLEEPY_FRAMES % 45 != 0:  # Only if not already counted
                    sleepy_eyes_counter += 1
                    print(f"[SLEEPY] Total: {sleepy_eyes_counter}")
            
            EYE_CLOSED_FRAMES = 0
            SLEEPY_FRAMES = 0
        
        # === YAWN DETECTION ===
        if mar_value > MAR_YAWN_THRESHOLD:
            YAWN_FRAMES += 1
            mouth_color = (0, 0, 255)  # RED
            
            # Count yawn if sustained and cooldown passed
            if YAWN_FRAMES == YAWN_CONSECUTIVE_THRESHOLD and current_time - last_yawn_time > YAWN_COOLDOWN:
                yawning_detected = True
                yawns_counter += 1
                last_yawn_time = current_time
                print(f"[YAWN] Total: {yawns_counter}")
            elif YAWN_FRAMES > YAWN_CONSECUTIVE_THRESHOLD:
                yawning_detected = True
        elif mar_value > 0.4:
            mouth_color = (0, 255, 255)  # YELLOW
            YAWN_FRAMES = 0
        else:
            mouth_color = (0, 255, 0)  # GREEN
            YAWN_FRAMES = 0
        
        # Draw facial landmarks with FILLED circles
        for (x_coord, y_coord) in leftEye:
            cv2.circle(frame, (x_coord, y_coord), 3, eye_color, -1)
        for (x_coord, y_coord) in rightEye:
            cv2.circle(frame, (x_coord, y_coord), 3, eye_color, -1)
        for (x_coord, y_coord) in mouth:
            cv2.circle(frame, (x_coord, y_coord), 2, mouth_color, -1)
        
        # Draw convex hulls
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        
        cv2.drawContours(frame, [leftEyeHull], -1, eye_color, 2)
        cv2.drawContours(frame, [rightEyeHull], -1, eye_color, 2)
        cv2.drawContours(frame, [mouthHull], -1, mouth_color, 2)
        
        # Fill eyes with semi-transparent color
        overlay = frame.copy()
        cv2.fillPoly(overlay, [leftEyeHull], eye_color)
        cv2.fillPoly(overlay, [rightEyeHull], eye_color)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw face box
        cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), (0, 255, 0), 2)
        
        # Draw head tracking point
        cv2.circle(frame, (int(shape[30][0]), head_y), 5, (255, 0, 255), -1)
    
    # Alarm control
    if drowsiness_alert or yawning_detected or head_nodding_detected:
        play_alarm()
    else:
        stop_alarm()
    
    # Create stats panel (LEFT SIDE)
    stats_panel = np.zeros((h, 260, 3), dtype=np.uint8)
    cv2.rectangle(stats_panel, (0, 0), (260, 220), (50, 50, 50), -1)
    
    cv2.putText(stats_panel, "DETECTION STATS", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(stats_panel, f"Eyes Closed: {eyes_closed_counter}", (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 100), 2)
    cv2.putText(stats_panel, f"Sleepy Eyes: {sleepy_eyes_counter}", (15, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 255), 2)
    cv2.putText(stats_panel, f"Total Blinks: {total_blinks}", (15, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 2)
    cv2.putText(stats_panel, f"Yawns: {yawns_counter}", (15, 175),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 100), 2)
    cv2.putText(stats_panel, f"Head Nods: {head_nods_counter}", (15, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 150, 255), 2)
    
    # Eye state monitor
    monitor_color = (0, 100, 0) if eye_status == "OPEN" else (100, 100, 0) if eye_status == "SLEEPY" else (0, 0, 100)
    cv2.rectangle(stats_panel, (0, 230), (260, 400), monitor_color, -1)
    cv2.putText(stats_panel, "EYE STATE MONITOR", (15, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(stats_panel, f"Status: {eye_status}", (15, 295),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(stats_panel, f"Left EAR: {ear_left:.3f}", (15, 325),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(stats_panel, f"Right EAR: {ear_right:.3f}", (15, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    # Debug info
    cv2.putText(stats_panel, f"Closed: {EYE_CLOSED_FRAMES}f", (15, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(stats_panel, f"Sleepy: {SLEEPY_FRAMES}f", (130, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Head state
    cv2.putText(stats_panel, f"Head: {head_movement_state[:7]}", (15, 395),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    
    # Create right side panel
    right_panel = np.zeros((h, 320, 3), dtype=np.uint8)
    
    # Status panel (TOP RIGHT)
    status_color = (0, 255, 0) if eye_status == "OPEN" else (0, 0, 255) if eye_status == "CLOSED" else (0, 255, 255)
    cv2.rectangle(right_panel, (0, 0), (320, 80), status_color, -1)
    cv2.putText(right_panel, eye_status, (60, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    
    # MAR panel
    mar_color = (0, 255, 0) if mar_value < 0.4 else (0, 255, 255) if mar_value < MAR_YAWN_THRESHOLD else (0, 0, 255)
    cv2.rectangle(right_panel, (0, 85), (320, 155), mar_color, -1)
    cv2.putText(right_panel, f"MAR: {mar_value:.3f}", (50, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # EAR panel
    cv2.rectangle(right_panel, (0, 160), (320, 230), (80, 80, 80), -1)
    cv2.putText(right_panel, f"EAR: {((ear_left + ear_right) / 2):.3f}", (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Controls info
    cv2.rectangle(right_panel, (0, 235), (320, 310), (40, 40, 40), -1)
    cv2.putText(right_panel, "CONTROLS", (85, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(right_panel, "Q - Quit", (70, 285),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(right_panel, "R - Reset Counters", (40, 305),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Left eye display on frame
    cv2.putText(frame, f"Left Eye: {ear_left:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Alert banners on main frame
    if drowsiness_alert:
        cv2.rectangle(frame, (0, h - 80), (w, h - 20), (0, 0, 139), -1)
        cv2.putText(frame, "!!! DROWSINESS ALERT!!!", (w//2 - 180, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    if yawning_detected:
        cv2.rectangle(frame, (0, h - 140 if drowsiness_alert else h - 80), 
                     (w, h - 80 if drowsiness_alert else h - 20), (0, 255, 255), -1)
        cv2.putText(frame, "YAWNING DETECTED!", (w//2 - 150, h - (105 if drowsiness_alert else 45)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    if head_nodding_detected:
        y_pos = h - 200 if (drowsiness_alert and yawning_detected) else h - 140 if (drowsiness_alert or yawning_detected) else h - 80
        cv2.rectangle(frame, (0, y_pos), (w, y_pos + 60), (255, 0, 0), -1)
        cv2.putText(frame, "HEAD NODDING!", (w//2 - 130, y_pos + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Combine all panels
    final_frame = np.hstack([stats_panel, frame, right_panel])
    
    cv2.imshow("Drowsiness Detection System", final_frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == ord("Q"):
        print("\n[EXIT] Quitting...")
        break
    elif key == ord("r") or key == ord("R"):
        reset_counters()

stop_alarm()
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

print(f"\n=== SESSION SUMMARY ===")
print(f"Eyes Closed Events: {eyes_closed_counter}")
print(f"Sleepy Eyes Count: {sleepy_eyes_counter}")
print(f"Total Blinks: {total_blinks}")
print(f"Yawns Detected: {yawns_counter}")
print(f"Head Nods: {head_nods_counter}")