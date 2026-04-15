from ultralytics import YOLO
import cv2
import time
import serial

# ─── Serial Configuration ─────────────────────────────────────────
SERIAL_PORT = "COM3"  # Update this to your Arduino's serial port (e.g., 'COM3' on Windows or '/dev/ttyUSB0' on Linux)
BAUD_RATE   = 9600
# ──────────────────────────────────────────────────────────────────

def connect_serial(port, baud):
    """Establish serial connection with retry logic."""
    while True:
        try:
            ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)
            print(f"[Serial] Connected on {port} at {baud} baud")
            return ser
        except serial.SerialException as e:
            print(f"[Serial] Connection failed: {e}. Retrying in 3s...")
            time.sleep(3)

def send_serial(ser, signal: bytes):
    """Send signal to Arduino safely."""
    try:
        ser.write(signal)
        print(f"[Serial] Sent: {signal.decode()} -> Motor {'ON' if signal == b'1' else 'OFF'}")
    except serial.SerialException as e:
        print(f"[Serial] Write error: {e}")


class RealTimeTrafficLightSystem:
    def __init__(self, detection_time=45, camera_index=0):
        """
        Initialize the real-time traffic light system

        Args:
            detection_time: Time in seconds before making decision (default: 45)
            camera_index: Camera device index (default: 0 for main camera)
        """
        print("Loading YOLOv11 model...")
        self.model = YOLO('yolo11n.pt')

        self.detection_time = detection_time
        self.camera_index   = camera_index
        self.boundary_box   = None

        # Traffic light states
        self.pedestrian_light = "RED"
        self.vehicle_light    = "GREEN"

        # Serial connection
        self.ser = connect_serial(SERIAL_PORT, BAUD_RATE)

    # ── Boundary Box ──────────────────────────────────────────────

    def set_boundary_box(self, frame_width, frame_height):
        margin_x = int(frame_width  * 0.1)
        margin_y = int(frame_height * 0.1)
        self.boundary_box = {
            'x1': margin_x,
            'y1': margin_y,
            'x2': frame_width  - margin_x,
            'y2': frame_height - margin_y
        }

    def is_in_boundary(self, box):
        if self.boundary_box is None:
            return False
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        return (self.boundary_box['x1'] <= center_x <= self.boundary_box['x2'] and
                self.boundary_box['y1'] <= center_y <= self.boundary_box['y2'])

    # ── Green Light Duration ──────────────────────────────────────

    def calculate_green_light_time(self, count):
        if count == 0:
            return 0
        elif 1 <= count <= 5:
            return 15
        elif 6 <= count <= 10:
            return 20
        elif count > 10:
            return 90
        return 0

    # ── Draw Traffic Lights ───────────────────────────────────────

    def draw_traffic_lights(self, frame):
        height, width   = frame.shape[:2]
        light_width     = 80
        light_height    = 200
        margin          = 20
        circle_radius   = 25

        # Pedestrian Light (Left)
        ped_x        = margin
        ped_y        = height - light_height - margin
        ped_center_x = ped_x + light_width // 2
        red_y        = ped_y + 50
        green_y      = ped_y + 150

        cv2.rectangle(frame, (ped_x, ped_y),
                      (ped_x + light_width, ped_y + light_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (ped_x, ped_y),
                      (ped_x + light_width, ped_y + light_height), (255, 255, 255), 2)
        cv2.circle(frame, (ped_center_x, red_y),   circle_radius,
                   (0, 0, 255) if self.pedestrian_light == "RED"   else (50, 50, 50), -1)
        cv2.circle(frame, (ped_center_x, green_y), circle_radius,
                   (0, 255, 0) if self.pedestrian_light == "GREEN" else (50, 50, 50), -1)
        cv2.putText(frame, "PEDESTRIAN", (ped_x - 10, ped_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Vehicle Light (Right)
        veh_x        = width - light_width - margin
        veh_y        = height - light_height - margin
        veh_center_x = veh_x + light_width // 2

        cv2.rectangle(frame, (veh_x, veh_y),
                      (veh_x + light_width, veh_y + light_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (veh_x, veh_y),
                      (veh_x + light_width, veh_y + light_height), (255, 255, 255), 2)
        cv2.circle(frame, (veh_center_x, red_y),   circle_radius,
                   (0, 0, 255) if self.vehicle_light == "RED"   else (50, 50, 50), -1)
        cv2.circle(frame, (veh_center_x, green_y), circle_radius,
                   (0, 255, 0) if self.vehicle_light == "GREEN" else (50, 50, 50), -1)
        cv2.putText(frame, "VEHICLE", (veh_x + 5, veh_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return frame

    # ── Draw UI ───────────────────────────────────────────────────

    def draw_ui(self, frame, time_remaining, current_count, phase):
        height, width = frame.shape[:2]

        # Boundary box
        if self.boundary_box:
            box_color = (0, 255, 255) if current_count > 0 else (100, 100, 100)
            cv2.rectangle(frame,
                          (self.boundary_box['x1'], self.boundary_box['y1']),
                          (self.boundary_box['x2'], self.boundary_box['y2']),
                          box_color, 3)
            label = f"DETECTION ZONE - {current_count} PERSON(S) INSIDE"
            cv2.putText(frame, label,
                        (self.boundary_box['x1'] + 10, self.boundary_box['y1'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        # Top panel
        panel_height = 160
        cv2.rectangle(frame, (0, 0), (width, panel_height), (0, 0, 0), -1)

        if phase == 'counting':
            cv2.putText(frame, "PHASE: REAL-TIME DETECTION",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Decision in: {int(time_remaining)}s",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"People in Zone NOW: {current_count}",
                        (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

            if current_count == 0:
                next_action, action_color = "Vehicle stays GREEN",        (0, 255, 0)
            elif 1 <= current_count <= 4:
                next_action, action_color = "Pedestrian GREEN for 30s",   (255, 200, 0)
            elif 5 <= current_count <= 9:
                next_action, action_color = "Pedestrian GREEN for 60s",   (255, 150, 0)
            else:
                next_action, action_color = "Pedestrian GREEN for 90s",   (255, 100, 0)

            cv2.putText(frame, f"Next: {next_action}",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, action_color, 2)

        elif phase == 'pedestrian_green':
            cv2.putText(frame, "PHASE: PEDESTRIAN CROSSING",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Time Remaining: {int(time_remaining)}s",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Counted: {current_count} person(s)",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Pedestrians: WALK | Vehicles: STOP",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        elif phase == 'vehicle_green':
            cv2.putText(frame, "PHASE: VEHICLE FLOW",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "No Pedestrians Detected",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Vehicles: GO | Pedestrians: WAIT",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Next scan starting soon...",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.putText(frame, "Press 'Q' to quit | 'R' to restart",
                    (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        frame = self.draw_traffic_lights(frame)
        return frame

    # ── Main Loop ─────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"Error: Cannot access camera {self.camera_index}")
            return

        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read from camera")
            return

        height, width = frame.shape[:2]
        self.set_boundary_box(width, height)

        print("\n" + "="*60)
        print("REAL-TIME TRAFFIC LIGHT SYSTEM")
        print("="*60)
        print(f"Camera: Device {self.camera_index} | Resolution: {width}x{height}")
        print(f"Detection Time: {self.detection_time} seconds")
        print("\nTraffic Light + Motor Rules:")
        print("  NO pedestrians  : Vehicle GREEN | Motor OFF")
        print("  1-5 pedestrians : Pedestrian GREEN 15s | Motor ON")
        print("  6-10 pedestrians: Pedestrian GREEN 20s | Motor ON")
        print("  10+ pedestrians : Pedestrian GREEN 90s | Motor ON")
        print("="*60 + "\n")

        # Initial state
        self.vehicle_light    = "GREEN"
        self.pedestrian_light = "RED"
        send_serial(self.ser, b'0')   # Motor OFF at startup

        phase              = 'counting'
        start_time         = time.time()
        final_count        = 0
        greenlight_duration = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Warning: Failed to read frame")
                    break

                current_time = time.time()

                # ── PHASE: COUNTING ───────────────────────────────
                if phase == 'counting':
                    elapsed_time   = current_time - start_time
                    time_remaining = self.detection_time - elapsed_time

                    if time_remaining > 0:
                        results = self.model(frame, verbose=False, classes=[0], conf=0.5)
                        current_count = 0

                        if results[0].boxes is not None and len(results[0].boxes) > 0:
                            boxes       = results[0].boxes.xyxy.cpu().numpy()
                            classes     = results[0].boxes.cls.cpu().numpy()
                            confidences = results[0].boxes.conf.cpu().numpy()

                            for box, cls, conf in zip(boxes, classes, confidences):
                                if int(cls) == 0 and self.is_in_boundary(box):
                                    current_count += 1
                                    x1, y1, x2, y2 = map(int, box)
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(frame, f"Person #{current_count}",
                                                (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        final_count = current_count
                        frame = self.draw_ui(frame, time_remaining, current_count, phase)

                    else:
                        # ── Decision time ─────────────────────────
                        if final_count == 0:
                            # No pedestrians → keep vehicle green, motor OFF
                            print("\n[Decision] No pedestrians. Vehicle stays GREEN. Motor OFF.")
                            send_serial(self.ser, b'0')

                            self.vehicle_light    = "GREEN"
                            self.pedestrian_light = "RED"
                            phase      = 'vehicle_green'
                            start_time = time.time()

                        else:
                            # Pedestrians detected → switch lights, motor ON
                            greenlight_duration = self.calculate_green_light_time(final_count)
                            print(f"\n[Decision] {final_count} pedestrian(s). "
                                  f"Pedestrian GREEN for {greenlight_duration}s. Motor ON.")
                            send_serial(self.ser, b'1')

                            self.pedestrian_light = "GREEN"
                            self.vehicle_light    = "RED"
                            phase      = 'pedestrian_green'
                            start_time = time.time()

                # ── PHASE: PEDESTRIAN GREEN ───────────────────────
                elif phase == 'pedestrian_green':
                    elapsed_time   = current_time - start_time
                    time_remaining = greenlight_duration - elapsed_time

                    if time_remaining > 0:
                        frame = self.draw_ui(frame, time_remaining, final_count, phase)
                    else:
                        # Crossing done → back to vehicle green, motor OFF
                        print("\n[Phase] Pedestrian crossing done. Vehicle GREEN. Motor OFF.")
                        send_serial(self.ser, b'0')

                        self.pedestrian_light = "RED"
                        self.vehicle_light    = "GREEN"
                        phase       = 'counting'
                        start_time  = time.time()
                        final_count = 0

                # ── PHASE: VEHICLE GREEN (no pedestrians) ─────────
                elif phase == 'vehicle_green':
                    elapsed_time = current_time - start_time

                    if elapsed_time > 3:
                        # Resume counting after 3s pause
                        phase       = 'counting'
                        start_time  = time.time()
                        final_count = 0
                    else:
                        frame = self.draw_ui(frame, 0, 0, phase)

                # ── Display ───────────────────────────────────────
                cv2.imshow('Real-Time Traffic Light System - Camera Feed', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\nSystem stopped by user.")
                    break
                elif key == ord('r') or key == ord('R'):
                    print("\nRestarting system...")
                    send_serial(self.ser, b'0')
                    self.pedestrian_light = "RED"
                    self.vehicle_light    = "GREEN"
                    phase       = 'counting'
                    start_time  = time.time()
                    final_count = 0

        finally:
            print("\nShutting down...")
            send_serial(self.ser, b'0')   # Safety: motor OFF on exit
            self.ser.close()
            cap.release()
            cv2.destroyAllWindows()
            print("System terminated successfully.")


def main():
    print("="*60)
    print("REAL-TIME TRAFFIC LIGHT SYSTEM")
    print("Using YOLOv11 with COCO Pre-trained Model")
    print("="*60)

    DETECTION_TIME = 20   # Seconds before making decision
    CAMERA_INDEX   = 1    # 0 = main camera, 1 = secondary, etc.

    print(f"\nDetection Time : {DETECTION_TIME}s")
    print(f"Camera Device  : {CAMERA_INDEX}")
    print(f"Serial Port    : {SERIAL_PORT}\n")

    system = RealTimeTrafficLightSystem(
        detection_time=DETECTION_TIME,
        camera_index=CAMERA_INDEX
    )
    system.run()


if __name__ == "__main__":
    main()