from ultralytics import YOLO
import cv2
import time
import serial

# ─── Configuration ────────────────────────────────────────────────
# SERIAL_PORT = "/dev/ttyCH341USB0" #Jetson Nano
SERIAL_PORT = "COM4"                #Windows
BAUD_RATE        = 9600
DETECTION_TIME   = 10    # Counting phase duration (seconds)
CAMERA_INDEX     = 1
YELLOW_DURATION  = 5     # Must match Arduino YELLOW_DURATION
BUFFER_DURATION  = 5     # Must match Arduino YELLOW_DURATION
# ──────────────────────────────────────────────────────────────────

def connect_serial(port, baud):
    while True:
        try:
            ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)
            print(f"[Serial] Connected on {port} at {baud} baud")
            return ser
        except serial.SerialException as e:
            print(f"[Serial] Failed: {e}. Retrying in 3s...")
            time.sleep(3)

def send_serial(ser, signal: bytes):
    try:
        ser.write(signal)
        print(f"[Serial] Sent: '{signal.decode()}'")
    except serial.SerialException as e:
        print(f"[Serial] Write error: {e}")

class RealTimeTrafficLightSystem:
    def __init__(self):
        print("Loading YOLOv11 model...")
        self.model = YOLO('yolo11n.pt')
        self.model.to('cpu')
        print("Model loaded.\n")

        self.boundary_box     = None
        self.pedestrian_light = "RED"
        self.vehicle_light    = "GREEN"
        self.ser              = connect_serial(SERIAL_PORT, BAUD_RATE)

    # ── Boundary Box ──────────────────────────────────────────────

    def set_boundary_box(self, w, h):
        mx = int(w * 0.1)
        my = int(h * 0.1)
        self.boundary_box = {'x1': mx, 'y1': my, 'x2': w - mx, 'y2': h - my}

    def is_in_boundary(self, box):
        if self.boundary_box is None:
            return False
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        return (self.boundary_box['x1'] <= cx <= self.boundary_box['x2'] and
                self.boundary_box['y1'] <= cy <= self.boundary_box['y2'])

    # ── Timing Rules ──────────────────────────────────────────────

    def calculate_green_light_time(self, count):
        if   1 <= count <= 4: return 15
        elif 5 <= count <= 8: return 20
        elif count >= 9:      return 25
        return 0

    # ── Draw Traffic Lights ───────────────────────────────────────

    def draw_traffic_lights(self, frame):
        h, w        = frame.shape[:2]
        lw, lh      = 120, 40
        margin      = 10
        top_traffic = int(h * 0.04)
        r           = int(lh * 0.3125)

        # Pedestrian (Top) — RED left, GREEN right
        px  = w - lw - margin - 40      # shift left to make room for label
        py  = top_traffic
        pcy = py + lh // 2

        cv2.rectangle(frame, (px, py), (px + lw, py + lh), (50, 50, 50), -1)
        cv2.rectangle(frame, (px, py), (px + lw, py + lh), (255, 255, 255), 2)

        # Evenly divide lw into 3 slots: [gap][circle][gap][circle][gap]
        slot = lw // 3
        cv2.circle(frame, (px + slot,          pcy), r,
                   (0, 0, 255)   if self.pedestrian_light == "RED"   else (30, 30, 30), -1)
        cv2.circle(frame, (px + slot * 2,      pcy), r,
                   (0, 255, 0)   if self.pedestrian_light == "GREEN" else (30, 30, 30), -1)

        cv2.putText(frame, "PED", (px + lw + 5, pcy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Vehicle (Bottom) — RED left, YELLOW middle, GREEN right
        vx  = w - lw - margin - 40      # same x as PED for alignment
        vy  = top_traffic + lh + margin
        vcy = vy + lh // 2

        cv2.rectangle(frame, (vx, vy), (vx + lw, vy + lh), (50, 50, 50), -1)
        cv2.rectangle(frame, (vx, vy), (vx + lw, vy + lh), (255, 255, 255), 2)

        slot = lw // 4
        cv2.circle(frame, (vx + slot,          vcy), r,
                (0, 0, 255)    if self.vehicle_light == "RED"    else (30, 30, 30), -1)
        cv2.circle(frame, (vx + slot * 2,      vcy), r,
                   (0, 255, 255)  if self.vehicle_light == "YELLOW" else (30, 30, 30), -1)
        cv2.circle(frame, (vx + slot * 3,      vcy), r,
                   (0, 255, 0)    if self.vehicle_light == "GREEN"  else (30, 30, 30), -1)

        cv2.putText(frame, "VEH", (vx + lw + 5, vcy + 5),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame

    # ── Draw UI ───────────────────────────────────────────────────

    def draw_ui(self, frame, time_remaining, count, phase):
        h, w = frame.shape[:2]

        # Boundary box
        if self.boundary_box:
            bc = (0,255,255) if count > 0 else (100,100,100)
            cv2.rectangle(frame,
                          (self.boundary_box['x1'], self.boundary_box['y1']),
                          (self.boundary_box['x2'], self.boundary_box['y2']), bc, 3)
            cv2.putText(frame, f"DETECTION ZONE - {count} PERSON(S)",
                        (self.boundary_box['x1']+10, self.boundary_box['y1']-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, bc, 2)

        # Top panel
        cv2.rectangle(frame, (0,0), (w,110), (0,0,0), -1)

        if phase == 'counting':
            cv2.putText(frame, "PHASE: DETECTION",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
            cv2.putText(frame, f"People in Zone: {count}",
                        (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            if   count == 0:         na, nc = "Vehicle stays GREEN",      (0,255,0)
            elif 1 <= count <= 4:    na, nc = "Pedestrian GREEN for 15s", (255,200,0)
            elif 5 <= count <= 8:    na, nc = "Pedestrian GREEN for 20s", (255,150,0)
            else:                    na, nc = "Pedestrian GREEN for 25s", (255,100,0)
            cv2.putText(frame, f"Next: {na}",
                        (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, nc, 1)
            cv2.putText(frame, f"Decision in: {int(time_remaining)}s",
                        (10,105), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        elif phase == 'vehicle_yellow':
            cv2.putText(frame, "PHASE: VEHICLE YELLOW",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
            cv2.putText(frame, "Vehicles: SLOW DOWN | Pedestrians: WAIT",
                        (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(frame, f"Pedestrian crossing in: {int(time_remaining)}s",
                        (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            

        elif phase == 'pedestrian_green':
            cv2.putText(frame, "PHASE: PEDESTRIAN CROSSING",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
            cv2.putText(frame, f"Group size: {count} person(s)",
                        (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            cv2.putText(frame, "Pedestrians: WALK | Vehicles: STOP",
                        (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
            cv2.putText(frame, f"Time Remaining: {int(time_remaining)}s",
                        (10,105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        elif phase == 'pedestrian_buffer':
            cv2.putText(frame, "PHASE: BUFFER",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
            cv2.putText(frame, "Pedestrians: CLEAR ROAD | Vehicles: WAIT",
                        (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            cv2.putText(frame, f"Vehicle GREEN in: {int(time_remaining)}s",
                        (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        elif phase == 'vehicle_green':
            cv2.putText(frame, "PHASE: VEHICLE FLOW",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
            cv2.putText(frame, "No pedestrians detected",
                        (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
            cv2.putText(frame, "Vehicles: GO | Pedestrians: WAIT",
                        (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        cv2.putText(frame, "Q = Quit | R = Restart",
                    (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        frame = self.draw_traffic_lights(frame)
        return frame

    # ── Main Run Loop ─────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {CAMERA_INDEX}")
            return

        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read from camera")
            return

        h, w = frame.shape[:2]
        self.set_boundary_box(w, h)

        print("="*60)
        print(f"Camera {CAMERA_INDEX} | {w}x{h} | Detection: {DETECTION_TIME}s")
        print("Rules: 1-4 ped=15s | 5-8 ped=20s | 9+ ped=25s")
        print("="*60 + "\n")

        # ── Initial state: Vehicle GREEN, Pedestrian RED ──────────
        self.vehicle_light    = "GREEN"
        self.pedestrian_light = "RED"
        send_serial(self.ser, b'0')

        phase              = 'counting'
        start_time         = time.time()
        final_count        = 0
        greenlight_duration = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                now = time.time()

                # ══ COUNTING ══════════════════════════════════════
                if phase == 'counting':
                    elapsed        = now - start_time
                    time_remaining = DETECTION_TIME - elapsed

                    if time_remaining > 0:
                        results       = self.model(frame, verbose=False,
                                                   classes=[0], conf=0.5)
                        current_count = 0

                        if (results[0].boxes is not None and
                                len(results[0].boxes) > 0):
                            boxes   = results[0].boxes.xyxy.cpu().numpy()
                            classes = results[0].boxes.cls.cpu().numpy()
                            for box, cls in zip(boxes, classes):
                                if int(cls) == 0 and self.is_in_boundary(box):
                                    current_count += 1
                                    x1,y1,x2,y2 = map(int, box)
                                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                                    cv2.putText(frame, f"#{current_count}",
                                                (x1, y1-10),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6, (0,255,0), 2)

                        final_count = current_count
                        frame = self.draw_ui(frame, time_remaining,
                                             current_count, phase)

                    else:
                        # ── Counting ended: make decision ─────────
                        if final_count == 0:
                            print("[Decision] No pedestrians → Vehicle GREEN")
                            self.vehicle_light    = "GREEN"
                            self.pedestrian_light = "RED"
                            phase      = 'vehicle_green'
                            start_time = now
                        else:
                            greenlight_duration = self.calculate_green_light_time(final_count)
                            print(f"\n[Decision] Count={final_count} → Duration={greenlight_duration}s")
                            send_serial(self.ser, b'1')

                            print(f"[Decision] {final_count} person(s) → "
                                  f"Send '1', Yellow 5s, "
                                  f"Ped GREEN {greenlight_duration}s")
                            send_serial(self.ser, b'1')
                            self.vehicle_light    = "YELLOW"
                            self.pedestrian_light = "RED"
                            phase      = 'vehicle_yellow'
                            start_time = now

                # ══ VEHICLE YELLOW (5s) ═══════════════════════════
                elif phase == 'vehicle_yellow':
                    elapsed        = now - start_time
                    time_remaining = YELLOW_DURATION - elapsed

                    if time_remaining > 0:
                        frame = self.draw_ui(frame, time_remaining,
                                             final_count, phase)
                    else:
                        print(f"[Phase] Yellow done → Ped GREEN {greenlight_duration}s")
                        self.vehicle_light    = "RED"
                        self.pedestrian_light = "GREEN"
                        phase      = 'pedestrian_green'
                        start_time = now

                # ══ PEDESTRIAN GREEN (volume-based) ═══════════════
                elif phase == 'pedestrian_green':
                    elapsed        = now - start_time
                    time_remaining = greenlight_duration - elapsed

                    print(f"[Debug] Ped GREEN | Count: {final_count} | "
                          f"Duration: {greenlight_duration}s | "
                          f"Remaining: {int(time_remaining)}s", end='\r')

                    if time_remaining > 0:
                        frame = self.draw_ui(frame, time_remaining,
                                             final_count, phase)
                    else:
                        print("[Phase] Crossing done → Send '0', Buffer 5s")
                        send_serial(self.ser, b'0')
                        self.vehicle_light    = "RED"
                        self.pedestrian_light = "RED"
                        phase      = 'pedestrian_buffer'
                        start_time = now

                # ══ PEDESTRIAN BUFFER (5s) ════════════════════════
                elif phase == 'pedestrian_buffer':
                    elapsed        = now - start_time
                    time_remaining = BUFFER_DURATION - elapsed

                    if time_remaining > 0:
                        frame = self.draw_ui(frame, time_remaining,
                                             final_count, phase)
                    else:
                        print("[Phase] Buffer done → Vehicle GREEN, restart count")
                        self.vehicle_light    = "GREEN"
                        self.pedestrian_light = "RED"
                        phase       = 'counting'
                        start_time  = now
                        final_count = 0

                # ══ VEHICLE GREEN (no pedestrians, 3s pause) ══════
                elif phase == 'vehicle_green':
                    elapsed = now - start_time
                    if elapsed > 3:
                        phase       = 'counting'
                        start_time  = now
                        final_count = 0
                    else:
                        frame = self.draw_ui(frame, 0, 0, phase)

                # ── Display ───────────────────────────────────────
                cv2.imshow('Traffic Light System', frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == ord('Q'):
                    print("\nStopped by user.")
                    break
                elif key == ord('r') or key == ord('R'):
                    print("\nRestarting...")
                    send_serial(self.ser, b'0')
                    self.vehicle_light    = "GREEN"
                    self.pedestrian_light = "RED"
                    phase       = 'counting'
                    start_time  = now
                    final_count = 0

        finally:
            print("\nShutting down...")
            send_serial(self.ser, b'0')
            self.ser.close()
            cap.release()
            cv2.destroyAllWindows()
            print("Done.")


if __name__ == "__main__":
    system = RealTimeTrafficLightSystem()
    system.run()