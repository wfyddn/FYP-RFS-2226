from ultralytics import YOLO
import cv2
import time
import numpy as np
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QFont


# ──────────────────────────────────────────────────────────────
#  Traffic light bulb widget
# ──────────────────────────────────────────────────────────────
class TrafficLightWidget(QWidget):
    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.label = label
        self.state = "RED"          # "RED" | "YELLOW" | "GREEN"
        self.setFixedSize(90, 230)

    def setState(self, state):
        if self.state != state:
            self.state = state
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # housing
        p.setBrush(QBrush(QColor("#1a1a1a")))
        p.setPen(QPen(QColor("#444444"), 2))
        p.drawRoundedRect(10, 5, 70, 195, 10, 10)

        bulbs = [
            ("RED",    "#ff2222", "#2a0808", 48),
            ("YELLOW", "#ffaa00", "#221500", 103),
            ("GREEN",  "#22cc44", "#082010", 158),
        ]
        for name, on_col, off_col, cy in bulbs:
            color = on_col if self.state == name else off_col
            p.setBrush(QBrush(QColor(color)))
            p.setPen(Qt.NoPen)
            p.drawEllipse(25, cy - 20, 40, 40)

        # label
        p.setPen(QColor("#888888"))
        p.setFont(QFont("Arial", 8))
        p.drawText(0, 205, 90, 18, Qt.AlignCenter, self.label)


# ──────────────────────────────────────────────────────────────
#  Stat card
# ──────────────────────────────────────────────────────────────
class StatCard(QFrame):
    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background: #1e2530;
                border: 1px solid #2a3545;
                border-radius: 8px;
            }
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)

        self._lbl = QLabel(label.upper())
        self._lbl.setStyleSheet(
            "color:#4a6070; font-size:10px; letter-spacing:1px;"
            "background:transparent; border:none;"
        )
        self._val = QLabel("—")
        self._val.setFont(QFont("Courier New", 22, QFont.Bold))
        self._val.setStyleSheet("color:#e0e4ec; background:transparent; border:none;")

        lay.addWidget(self._lbl)
        lay.addWidget(self._val)

    def setValue(self, text, color="#e0e4ec"):
        self._val.setText(text)
        self._val.setStyleSheet(
            f"color:{color}; background:transparent; border:none;"
        )


# ──────────────────────────────────────────────────────────────
#  Detection + phase logic thread
# ──────────────────────────────────────────────────────────────
class DetectionThread(QThread):
    update_ui = pyqtSignal(dict)

    def __init__(self, detection_time=20, camera_index=0):
        super().__init__()
        self.detection_time = detection_time
        self.camera_index   = camera_index
        self._running       = True

        self.vehicle_light    = "GREEN"
        self.pedestrian_light = "RED"
        self.boundary_box     = None

        print("Loading YOLOv11 model...")
        self.model = YOLO('yolo11n.pt')

    # ── helpers (identical to original) ───────────────────────
    def set_boundary_box(self, w, h):
        mx = int(w * 0.1)
        my = int(h * 0.1)
        self.boundary_box = {'x1': mx, 'y1': my, 'x2': w - mx, 'y2': h - my}

    def is_in_boundary(self, box):
        if self.boundary_box is None:
            return False
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        b  = self.boundary_box
        return b['x1'] <= cx <= b['x2'] and b['y1'] <= cy <= b['y2']

    def calculate_green_light_time(self, count):
        if count == 0:  return 0
        if count <= 10:  return 25
        if count <= 20: return 30
        return 40

    # ── draw helpers for the OpenCV window ────────────────────
    def draw_traffic_lights(self, frame):
        """Identical to your original draw_traffic_lights."""
        height, width = frame.shape[:2]
        light_width   = 80
        light_height  = 200
        margin        = 20
        circle_radius = 25

        # Pedestrian (left)
        ped_x = margin
        ped_y = height - light_height - margin
        cv2.rectangle(frame, (ped_x, ped_y),
                      (ped_x + light_width, ped_y + light_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (ped_x, ped_y),
                      (ped_x + light_width, ped_y + light_height), (255, 255, 255), 2)
        ped_cx = ped_x + light_width // 2
        red_y  = ped_y + 50
        grn_y  = ped_y + 150
        cv2.circle(frame, (ped_cx, red_y), circle_radius,
                   (0, 0, 255) if self.pedestrian_light == "RED" else (50, 50, 50), -1)
        cv2.circle(frame, (ped_cx, grn_y), circle_radius,
                   (0, 255, 0) if self.pedestrian_light == "GREEN" else (50, 50, 50), -1)
        cv2.putText(frame, "PEDESTRIAN", (ped_x - 10, ped_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Vehicle (right)
        veh_x  = width - light_width - margin
        veh_y  = height - light_height - margin
        cv2.rectangle(frame, (veh_x, veh_y),
                      (veh_x + light_width, veh_y + light_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (veh_x, veh_y),
                      (veh_x + light_width, veh_y + light_height), (255, 255, 255), 2)
        veh_cx = veh_x + light_width // 2
        cv2.circle(frame, (veh_cx, red_y), circle_radius,
                   (0, 0, 255) if self.vehicle_light == "RED" else (50, 50, 50), -1)
        cv2.circle(frame, (veh_cx, grn_y), circle_radius,
                   (0, 255, 0) if self.vehicle_light == "GREEN" else (50, 50, 50), -1)
        cv2.putText(frame, "VEHICLE", (veh_x + 5, veh_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return frame

    def draw_ui(self, frame, time_remaining, current_count, phase):
        """Identical to your original draw_ui."""
        height, width = frame.shape[:2]

        if self.boundary_box:
            box_color = (0, 255, 255) if current_count > 0 else (100, 100, 100)
            cv2.rectangle(frame,
                          (self.boundary_box['x1'], self.boundary_box['y1']),
                          (self.boundary_box['x2'], self.boundary_box['y2']),
                          box_color, 3)
            cv2.putText(frame,
                        f"DETECTION ZONE - {current_count} PERSON(S) INSIDE",
                        (self.boundary_box['x1'] + 10, self.boundary_box['y1'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

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
                next_action, action_color = "Vehicle stays GREEN", (0, 255, 0)
            elif current_count <= 4:
                next_action, action_color = "Pedestrian GREEN for 15s", (255, 200, 0)
            elif current_count <= 9:
                next_action, action_color = "Pedestrian GREEN for 20s", (255, 150, 0)
            else:
                next_action, action_color = "Pedestrian GREEN for 90s", (255, 100, 0)
            cv2.putText(frame, f"Next: {next_action}",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, action_color, 2)

        elif phase == 'yellow':
            cv2.putText(frame, "PHASE: YELLOW TRANSITION",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.putText(frame, f"Switching in: {time_remaining:.1f}s",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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

    def stop(self):
        self._running = False

    # ── main loop ──────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"Error: Cannot access camera {self.camera_index}")
            return

        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read from camera")
            return

        h, w = frame.shape[:2]
        self.set_boundary_box(w, h)

        print("\n" + "="*60)
        print("REAL-TIME TRAFFIC LIGHT SYSTEM")
        print("="*60)

        phase               = 'counting'
        start_time          = time.time()
        final_count         = 0
        greenlight_duration = 0
        YELLOW_GAP          = 3.0

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('r') or key == ord('R'):
                self.vehicle_light    = "GREEN"
                self.pedestrian_light = "RED"
                phase      = 'counting'
                start_time = time.time()
                final_count = 0

            # ── counting ──────────────────────────────────────
            if phase == 'counting':
                elapsed        = now - start_time
                time_remaining = self.detection_time - elapsed

                if time_remaining > 0:
                    results = self.model(frame, verbose=False, classes=[0], conf=0.5)
                    current_count = 0

                    if results[0].boxes is not None and len(results[0].boxes) > 0:
                        boxes   = results[0].boxes.xyxy.cpu().numpy()
                        classes = results[0].boxes.cls.cpu().numpy()
                        confs   = results[0].boxes.conf.cpu().numpy()

                        for box, cls, conf in zip(boxes, classes, confs):
                            if int(cls) == 0 and self.is_in_boundary(box):
                                current_count += 1
                                x1, y1, x2, y2 = map(int, box)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"Person #{current_count}",
                                            (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    final_count = current_count
                    frame = self.draw_ui(frame, time_remaining, current_count, 'counting')

                    self.update_ui.emit({
                        'phase':     'counting',
                        'timer':     time_remaining,
                        'count':     current_count,
                        'green_dur': self.calculate_green_light_time(current_count),
                        'veh':       self.vehicle_light,
                        'ped':       self.pedestrian_light,
                    })

                else:
                    # yellow gap
                    self.vehicle_light    = "YELLOW"
                    self.pedestrian_light = "RED"
                    y_start = time.time()

                    while time.time() - y_start < YELLOW_GAP and self._running:
                        rem = YELLOW_GAP - (time.time() - y_start)
                        ret2, f2 = cap.read()
                        if ret2:
                            f2 = self.draw_ui(f2, rem, final_count, 'yellow')
                            cv2.imshow('Real-Time Traffic Light System - Camera Feed', f2)
                            cv2.waitKey(1)
                        self.update_ui.emit({
                            'phase': 'yellow', 'timer': rem,
                            'count': final_count,
                            'green_dur': self.calculate_green_light_time(final_count),
                            'veh': self.vehicle_light, 'ped': self.pedestrian_light,
                        })
                        time.sleep(0.05)

                    if final_count == 0:
                        self.vehicle_light    = "GREEN"
                        self.pedestrian_light = "RED"
                        phase      = 'vehicle_green'
                        start_time = time.time()
                    else:
                        greenlight_duration   = self.calculate_green_light_time(final_count)
                        self.vehicle_light    = "RED"
                        self.pedestrian_light = "GREEN"
                        phase      = 'pedestrian_green'
                        start_time = time.time()

            # ── pedestrian green ──────────────────────────────
            elif phase == 'pedestrian_green':
                elapsed        = now - start_time
                time_remaining = greenlight_duration - elapsed

                if time_remaining > 0:
                    frame = self.draw_ui(frame, time_remaining, final_count, 'pedestrian_green')
                    self.update_ui.emit({
                        'phase':     'pedestrian_green',
                        'timer':     time_remaining,
                        'count':     final_count,
                        'green_dur': greenlight_duration,
                        'veh':       self.vehicle_light,
                        'ped':       self.pedestrian_light,
                    })
                else:
                    # yellow gap back
                    self.vehicle_light    = "RED"
                    self.pedestrian_light = "YELLOW"
                    y_start = time.time()

                    while time.time() - y_start < YELLOW_GAP and self._running:
                        rem = YELLOW_GAP - (time.time() - y_start)
                        ret2, f2 = cap.read()
                        if ret2:
                            f2 = self.draw_ui(f2, rem, final_count, 'yellow')
                            cv2.imshow('Real-Time Traffic Light System - Camera Feed', f2)
                            cv2.waitKey(1)
                        self.update_ui.emit({
                            'phase': 'yellow', 'timer': rem,
                            'count': final_count,
                            'green_dur': greenlight_duration,
                            'veh': self.vehicle_light, 'ped': self.pedestrian_light,
                        })
                        time.sleep(0.05)

                    self.vehicle_light    = "GREEN"
                    self.pedestrian_light = "RED"
                    phase       = 'counting'
                    start_time  = time.time()
                    final_count = 0

            # ── vehicle green (no pedestrians) ────────────────
            elif phase == 'vehicle_green':
                elapsed = now - start_time
                frame = self.draw_ui(frame, 0, 0, 'vehicle_green')
                self.update_ui.emit({
                    'phase': 'vehicle_green', 'timer': max(0, 3.0 - elapsed),
                    'count': 0, 'green_dur': 0,
                    'veh': self.vehicle_light, 'ped': self.pedestrian_light,
                })
                if elapsed > 3.0:
                    phase      = 'counting'
                    start_time = time.time()
                    final_count = 0

            cv2.imshow('Real-Time Traffic Light System - Camera Feed', frame)

        cap.release()
        cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────
#  PyQt5 GUI window
# ──────────────────────────────────────────────────────────────
class GUIWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Light Monitor")
        self.setStyleSheet("QMainWindow { background: #111620; }")
        self.setFixedSize(360, 420)

        central = QWidget()
        central.setStyleSheet("background: #111620;")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        # ── traffic lights row ────────────────────────────────
        lights_row = QHBoxLayout()
        lights_row.setAlignment(Qt.AlignCenter)
        lights_row.setSpacing(30)

        self.veh_light = TrafficLightWidget("Vehicle")
        self.ped_light = TrafficLightWidget("Pedestrian")
        self.veh_light.setState("GREEN")
        self.ped_light.setState("RED")

        lights_row.addWidget(self.veh_light)
        lights_row.addWidget(self.ped_light)
        root.addLayout(lights_row)

        # ── phase label ───────────────────────────────────────
        self.phase_label = QLabel("Detecting pedestrians")
        self.phase_label.setAlignment(Qt.AlignCenter)
        self.phase_label.setStyleSheet(
            "color:#5a8aaf; font-size:11px; letter-spacing:1px;"
        )
        root.addWidget(self.phase_label)

        # ── stat cards ────────────────────────────────────────
        self.card_timer = StatCard("Detection timer")
        self.card_count = StatCard("Pedestrians in zone")
        self.card_green = StatCard("Pedestrian green duration")

        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self.card_timer)
        row.addWidget(self.card_count)
        root.addLayout(row)
        root.addWidget(self.card_green)

    def on_update(self, state):
        phase = state['phase']
        timer = state['timer']
        count = state['count']
        green = state['green_dur']

        self.veh_light.setState(state['veh'])
        self.ped_light.setState(state['ped'])

        if phase == 'counting':
            self.phase_label.setText("Detecting pedestrians")
            self.card_timer.setValue(f"{int(timer)}s", "#5baaf5")
            self.card_count.setValue(str(count), "#3ddc84" if count > 0 else "#e0e4ec")
            self.card_green.setValue(f"{green}s" if green > 0 else "—", "#f0a030")

        elif phase == 'yellow':
            self.phase_label.setText("Yellow transition — 3s gap")
            self.card_timer.setValue(f"{timer:.1f}s", "#ffaa00")

        elif phase == 'pedestrian_green':
            self.phase_label.setText("Pedestrian crossing")
            self.card_timer.setValue(f"{int(timer)}s", "#3ddc84")
            self.card_count.setValue(str(count), "#3ddc84")
            self.card_green.setValue(f"{green}s", "#3ddc84")

        elif phase == 'vehicle_green':
            self.phase_label.setText("No pedestrians — vehicle flow")
            self.card_timer.setValue("—", "#e0e4ec")
            self.card_count.setValue("0", "#e0e4ec")
            self.card_green.setValue("—", "#e0e4ec")

    def closeEvent(self, event):
        event.accept()


# ──────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────
def main():
    DETECTION_TIME = 30
    CAMERA_INDEX   = 0

    app = QApplication(sys.argv)

    # GUI window
    gui = GUIWindow()
    gui.show()

    # Detection thread
    thread = DetectionThread(detection_time=DETECTION_TIME, camera_index=CAMERA_INDEX)
    thread.update_ui.connect(gui.on_update)
    thread.start()

    # Clean shutdown
    def on_quit():
        thread.stop()
        thread.wait()

    app.aboutToQuit.connect(on_quit)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()