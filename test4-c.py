import sys
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QColor, QPainter, QBrush, QPen, QImage, QPixmap

from ultralytics import YOLO


# ──────────────────────────────────────────────────────────────
#  Detection thread  (runs YOLO + phase logic off the main thread)
# ──────────────────────────────────────────────────────────────
class DetectionThread(QThread):
    update_ui    = pyqtSignal(dict)
    update_frame = pyqtSignal(np.ndarray)

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

    def set_boundary_box(self, w, h):
        mx, my = int(w * 0.1), int(h * 0.1)
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
        if count <= 5:  return 15
        if count <= 10: return 20
        return 90

    def draw_boundary(self, frame, count):
        if self.boundary_box is None:
            return frame
        b     = self.boundary_box
        color = (0, 255, 255) if count > 0 else (100, 100, 100)
        cv2.rectangle(frame, (b['x1'], b['y1']), (b['x2'], b['y2']), color, 3)
        cv2.putText(frame,
                    f"DETECTION ZONE - {count} PERSON(S)",
                    (b['x1'] + 10, b['y1'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

    def stop(self):
        self._running = False

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_index}")
            return

        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read from camera")
            return

        h, w = frame.shape[:2]
        self.set_boundary_box(w, h)

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

            # ── counting phase ────────────────────────────────
            if phase == 'counting':
                elapsed        = now - start_time
                time_remaining = self.detection_time - elapsed

                if time_remaining > 0:
                    results = self.model(frame, verbose=False, classes=[0], conf=0.5)
                    current_count = 0

                    if results[0].boxes is not None and len(results[0].boxes) > 0:
                        boxes   = results[0].boxes.xyxy.cpu().numpy()
                        classes = results[0].boxes.cls.cpu().numpy()

                        for box, cls in zip(boxes, classes):
                            if int(cls) == 0 and self.is_in_boundary(box):
                                current_count += 1
                                x1, y1, x2, y2 = map(int, box)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"#{current_count}",
                                            (x1, y1 - 8),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    final_count = current_count
                    frame = self.draw_boundary(frame, current_count)
                    self.update_frame.emit(frame.copy())
                    self.update_ui.emit({
                        'phase':     'counting',
                        'timer':     time_remaining,
                        'count':     current_count,
                        'green_dur': self.calculate_green_light_time(current_count),
                        'veh_light': self.vehicle_light,
                        'ped_light': self.pedestrian_light,
                    })

                else:
                    # yellow gap before switching
                    self.vehicle_light    = "YELLOW"
                    self.pedestrian_light = "RED"
                    yellow_start = time.time()

                    while time.time() - yellow_start < YELLOW_GAP and self._running:
                        rem = YELLOW_GAP - (time.time() - yellow_start)
                        self.update_ui.emit({
                            'phase':     'yellow',
                            'timer':     rem,
                            'count':     final_count,
                            'green_dur': self.calculate_green_light_time(final_count),
                            'veh_light': self.vehicle_light,
                            'ped_light': self.pedestrian_light,
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

            # ── pedestrian green phase ────────────────────────
            elif phase == 'pedestrian_green':
                elapsed        = now - start_time
                time_remaining = greenlight_duration - elapsed

                if time_remaining > 0:
                    frame = self.draw_boundary(frame, final_count)
                    self.update_frame.emit(frame.copy())
                    self.update_ui.emit({
                        'phase':     'pedestrian_green',
                        'timer':     time_remaining,
                        'count':     final_count,
                        'green_dur': greenlight_duration,
                        'veh_light': self.vehicle_light,
                        'ped_light': self.pedestrian_light,
                    })
                else:
                    # yellow gap before switching back
                    self.vehicle_light    = "RED"
                    self.pedestrian_light = "YELLOW"
                    yellow_start = time.time()

                    while time.time() - yellow_start < YELLOW_GAP and self._running:
                        rem = YELLOW_GAP - (time.time() - yellow_start)
                        self.update_ui.emit({
                            'phase':     'yellow',
                            'timer':     rem,
                            'count':     final_count,
                            'green_dur': greenlight_duration,
                            'veh_light': self.vehicle_light,
                            'ped_light': self.pedestrian_light,
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
                frame = self.draw_boundary(frame, 0)
                self.update_frame.emit(frame.copy())
                self.update_ui.emit({
                    'phase':     'vehicle_green',
                    'timer':     max(0, 3.0 - elapsed),
                    'count':     0,
                    'green_dur': 0,
                    'veh_light': self.vehicle_light,
                    'ped_light': self.pedestrian_light,
                })
                if elapsed > 3.0:
                    phase      = 'counting'
                    start_time = time.time()

            time.sleep(0.01)

        cap.release()


# ──────────────────────────────────────────────────────────────
#  Traffic light widget
# ──────────────────────────────────────────────────────────────
class TrafficLightWidget(QWidget):
    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.label = label
        self.state = "RED"
        self.setFixedSize(80, 220)

    def setState(self, state):
        if self.state != state:
            self.state = state
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # housing
        p.setBrush(QBrush(QColor("#1a1a1a")))
        p.setPen(QPen(QColor("#333333"), 2))
        p.drawRoundedRect(5, 5, 70, 190, 10, 10)

        bulbs = [
            ("RED",    "#ff2222", "#2a0808", 45),
            ("YELLOW", "#ffaa00", "#221500", 100),
            ("GREEN",  "#22cc44", "#082010", 155),
        ]
        for name, on_color, off_color, cy in bulbs:
            color = on_color if self.state == name else off_color
            p.setBrush(QBrush(QColor(color)))
            p.setPen(Qt.NoPen)
            p.drawEllipse(20, cy - 20, 40, 40)

        # label below housing
        p.setPen(QColor("#666666"))
        p.setFont(QFont("Arial", 8))
        p.drawText(0, 200, 80, 18, Qt.AlignCenter, self.label)


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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)

        self._lbl = QLabel(label.upper())
        self._lbl.setStyleSheet(
            "color:#4a6070; font-size:10px; letter-spacing:1px;"
            "background:transparent; border:none;"
        )

        self._val = QLabel("—")
        self._val.setFont(QFont("Courier New", 22, QFont.Bold))
        self._val.setStyleSheet(
            "color:#e0e4ec; background:transparent; border:none;"
        )

        layout.addWidget(self._lbl)
        layout.addWidget(self._val)

    def setValue(self, text, color="#e0e4ec"):
        self._val.setText(text)
        self._val.setStyleSheet(
            f"color:{color}; background:transparent; border:none;"
        )


# ──────────────────────────────────────────────────────────────
#  Main window
# ──────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Light System")
        self.setStyleSheet("QMainWindow { background: #111620; }")
        self.setMinimumSize(860, 520)

        central = QWidget()
        central.setStyleSheet("background: #111620;")
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(20)

        # ── camera feed (left) ────────────────────────────────
        self.cam_label = QLabel("Camera initialising…")
        self.cam_label.setFixedSize(480, 360)
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setStyleSheet(
            "background:#0a0e14; border:1px solid #1e2a38;"
            "border-radius:8px; color:#334455; font-size:12px;"
        )
        root.addWidget(self.cam_label)

        # ── right panel ───────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(14)

        # traffic lights
        lights_row = QHBoxLayout()
        lights_row.setAlignment(Qt.AlignCenter)
        lights_row.setSpacing(30)
        self.veh_light = TrafficLightWidget("Vehicle")
        self.ped_light = TrafficLightWidget("Pedestrian")
        self.veh_light.setState("GREEN")
        self.ped_light.setState("RED")
        lights_row.addWidget(self.veh_light)
        lights_row.addWidget(self.ped_light)
        right.addLayout(lights_row)

        # phase label
        self.phase_label = QLabel("Detecting pedestrians")
        self.phase_label.setAlignment(Qt.AlignCenter)
        self.phase_label.setStyleSheet(
            "color:#5a8aaf; font-size:11px; letter-spacing:1px;"
        )
        right.addWidget(self.phase_label)

        # stat cards
        self.card_timer = StatCard("Detection timer")
        self.card_count = StatCard("Pedestrians in zone")
        self.card_green = StatCard("Pedestrian green duration")

        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self.card_timer)
        row.addWidget(self.card_count)
        right.addLayout(row)
        right.addWidget(self.card_green)
        right.addStretch()

        root.addLayout(right)

        # ── start thread ──────────────────────────────────────
        self.thread = DetectionThread(detection_time=20, camera_index=0)
        self.thread.update_ui.connect(self.on_update_ui)
        self.thread.update_frame.connect(self.on_update_frame)
        self.thread.start()

    # ── slots ─────────────────────────────────────────────────
    def on_update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.cam_label.width(), self.cam_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.cam_label.setPixmap(pix)

    def on_update_ui(self, state):
        phase = state['phase']
        timer = state['timer']
        count = state['count']
        green = state['green_dur']

        self.veh_light.setState(state['veh_light'])
        self.ped_light.setState(state['ped_light'])

        if phase == 'counting':
            self.phase_label.setText("Detecting pedestrians")
            self.card_timer.setValue(f"{int(timer)}s", "#5baaf5")
            self.card_count.setValue(str(count), "#3ddc84" if count > 0 else "#e0e4ec")
            self.card_green.setValue(f"{green}s" if green > 0 else "—", "#f0a030")

        elif phase == 'yellow':
            self.phase_label.setText("Yellow transition  —  3s gap")
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
        self.thread.stop()
        self.thread.wait()
        event.accept()


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())