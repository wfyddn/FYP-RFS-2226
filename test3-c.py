from ultralytics import YOLO
import cv2
import time
import numpy as np

class RealTimeTrafficLightSystem:
    def __init__(self, detection_time=45, camera_index=0):
        """
        Initialize the real-time traffic light system
        
        Args:
            detection_time: Time in seconds before making decision (default: 45)
            camera_index: Camera device index (default: 0 for main camera)
        """
        # Load YOLOv11 model (pretrained on COCO dataset)
        print("Loading YOLOv11 model...")
        self.model = YOLO('yolo11n.pt')
        
        self.detection_time = detection_time
        self.camera_index = camera_index
        
        # Define boundary box
        self.boundary_box = None
        
        # Traffic light states
        self.pedestrian_light = "RED"
        self.vehicle_light = "GREEN"
        
    def set_boundary_box(self, frame_width, frame_height):
        """Set the boundary box for counting"""
        margin_x = int(frame_width * 0.1)
        margin_y = int(frame_height * 0.1)
        
        self.boundary_box = {
            'x1': margin_x,
            'y1': margin_y,
            'x2': frame_width - margin_x,
            'y2': frame_height - margin_y
        }
        
    def is_in_boundary(self, box):
        """Check if the detected person's center is within boundary box"""
        if self.boundary_box is None:
            return False
        
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        
        return (self.boundary_box['x1'] <= center_x <= self.boundary_box['x2'] and
                self.boundary_box['y1'] <= center_y <= self.boundary_box['y2'])
    
    def calculate_green_light_time(self, count):
        """
        Calculate pedestrian green light duration based on current pedestrian count
        
        Args:
            count: Number of pedestrians currently in the zone
            
        Returns:
            Green light duration in seconds (0 if no pedestrians)
        """
        if count == 0:
            return 0
        elif 1 <= count <= 5:
            return 15
        elif 6 <= count <= 10:
            return 20
        elif count > 10:
            return 90
        else:
            return 0
    
    def draw_traffic_lights(self, frame):
        """Draw both traffic light signals on the frame"""
        height, width = frame.shape[:2]
        
        # Traffic light dimensions
        light_width = 80
        light_height = 200
        margin = 20
        circle_radius = 25
        
        # Pedestrian Light (Left side)
        ped_x = margin
        ped_y = height - light_height - margin
        
        # Draw pedestrian light box
        cv2.rectangle(frame, (ped_x, ped_y), 
                     (ped_x + light_width, ped_y + light_height),
                     (50, 50, 50), -1)
        cv2.rectangle(frame, (ped_x, ped_y), 
                     (ped_x + light_width, ped_y + light_height),
                     (255, 255, 255), 2)
        
        # Pedestrian light circles
        ped_center_x = ped_x + light_width // 2
        red_y = ped_y + 50
        green_y = ped_y + 150
        
        # Red circle (top)
        if self.pedestrian_light == "RED":
            cv2.circle(frame, (ped_center_x, red_y), circle_radius, (0, 0, 255), -1)
        else:
            cv2.circle(frame, (ped_center_x, red_y), circle_radius, (50, 50, 50), -1)
        
        # Green circle (bottom)
        if self.pedestrian_light == "GREEN":
            cv2.circle(frame, (ped_center_x, green_y), circle_radius, (0, 255, 0), -1)
        else:
            cv2.circle(frame, (ped_center_x, green_y), circle_radius, (50, 50, 50), -1)
        
        # Pedestrian label
        cv2.putText(frame, "PEDESTRIAN", (ped_x - 10, ped_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Vehicle Light (Right side)
        veh_x = width - light_width - margin
        veh_y = height - light_height - margin
        
        # Draw vehicle light box
        cv2.rectangle(frame, (veh_x, veh_y), 
                     (veh_x + light_width, veh_y + light_height),
                     (50, 50, 50), -1)
        cv2.rectangle(frame, (veh_x, veh_y), 
                     (veh_x + light_width, veh_y + light_height),
                     (255, 255, 255), 2)
        
        # Vehicle light circles
        veh_center_x = veh_x + light_width // 2
        
        # Red circle (top)
        if self.vehicle_light == "RED":
            cv2.circle(frame, (veh_center_x, red_y), circle_radius, (0, 0, 255), -1)
        else:
            cv2.circle(frame, (veh_center_x, red_y), circle_radius, (50, 50, 50), -1)
        
        # Green circle (bottom)
        if self.vehicle_light == "GREEN":
            cv2.circle(frame, (veh_center_x, green_y), circle_radius, (0, 255, 0), -1)
        else:
            cv2.circle(frame, (veh_center_x, green_y), circle_radius, (50, 50, 50), -1)
        
        # Vehicle label
        cv2.putText(frame, "VEHICLE", (veh_x + 5, veh_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def draw_ui(self, frame, time_remaining, current_count, phase):
        """
        Draw UI elements on frame
        
        Args:
            frame: Video frame
            time_remaining: Time remaining in current phase
            current_count: Current number of people in zone RIGHT NOW
            phase: Current phase
        """
        height, width = frame.shape[:2]
        
        # Draw boundary box
        if self.boundary_box:
            # Change color based on count
            box_color = (0, 255, 255) if current_count > 0 else (100, 100, 100)
            cv2.rectangle(frame, 
                         (self.boundary_box['x1'], self.boundary_box['y1']),
                         (self.boundary_box['x2'], self.boundary_box['y2']),
                         box_color, 3)
            
            # Show zone label with current count
            label = f"DETECTION ZONE - {current_count} PERSON(S) INSIDE"
            cv2.putText(frame, label, 
                       (self.boundary_box['x1'] + 10, self.boundary_box['y1'] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
        
        # Create info panel at the top
        panel_height = 160
        cv2.rectangle(frame, (0, 0), (width, panel_height), (0, 0, 0), -1)
        
        if phase == 'counting':
            # Counting phase UI
            cv2.putText(frame, "PHASE: REAL-TIME DETECTION", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Decision in: {int(time_remaining)}s", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"People in Zone NOW: {current_count}", 
                       (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
            
            # Show what will happen
            if current_count == 0:
                next_action = "Vehicle stays GREEN"
                action_color = (0, 255, 0)
            elif 1 <= current_count <= 4:
                next_action = "Pedestrian GREEN for 30s"
                action_color = (255, 200, 0)
            elif 5 <= current_count <= 9:
                next_action = "Pedestrian GREEN for 60s"
                action_color = (255, 150, 0)
            else:
                next_action = "Pedestrian GREEN for 90s"
                action_color = (255, 100, 0)
            
            cv2.putText(frame, f"Next: {next_action}", 
                       (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, action_color, 2)
            
        elif phase == 'pedestrian_green':
            # Pedestrian green light phase UI
            cv2.putText(frame, "PHASE: PEDESTRIAN CROSSING", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Time Remaining: {int(time_remaining)}s", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Counted: {current_count} person(s)", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Pedestrians: WALK | Vehicles: STOP", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        elif phase == 'vehicle_green':
            # Vehicle green light phase UI
            cv2.putText(frame, "PHASE: VEHICLE FLOW", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "No Pedestrians Detected", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Vehicles: GO | Pedestrians: WAIT", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Next scan starting soon...", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Instructions
        cv2.putText(frame, "Press 'Q' to quit | 'R' to restart", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw traffic lights
        frame = self.draw_traffic_lights(frame)
        
        return frame
    
    def run(self):
        """Run the real-time traffic light system using camera"""
        
        # Open camera
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            print(f"Error: Cannot access camera {self.camera_index}")
            print("Try changing camera_index in the code (0, 1, 2, etc.)")
            return
        
        # Get frame dimensions
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read from camera")
            return
        
        height, width = frame.shape[:2]
        self.set_boundary_box(width, height)
        
        print("\n" + "="*60)
        print("REAL-TIME TRAFFIC LIGHT SYSTEM")
        print("="*60)
        print(f"Camera: Device {self.camera_index}")
        print(f"Resolution: {width}x{height}")
        print(f"Detection Time: {self.detection_time} seconds")
        print("\nHow it works:")
        print("  - Camera detects people CURRENTLY in the zone")
        print("  - After 45 seconds, uses CURRENT count for decision")
        print("  - Count updates in REAL-TIME as people enter/leave")
        print("\nTraffic Light Rules (based on current count):")
        print("  NO pedestrians    : Vehicle GREEN, Pedestrian RED")
        print("  1-4 pedestrians   : Pedestrian GREEN (30s), Vehicle RED")
        print("  5-9 pedestrians   : Pedestrian GREEN (60s), Vehicle RED")
        print("  10+ pedestrians   : Pedestrian GREEN (90s), Vehicle RED")
        print("="*60)
        print("\nSystem started! Press 'Q' to quit, 'R' to restart\n")
        
        # Initial state: Vehicle GREEN, Pedestrian RED
        self.vehicle_light = "GREEN"
        self.pedestrian_light = "RED"
        
        # Start counting phase
        start_time = time.time()
        phase = 'counting'
        final_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame from camera")
                break
            
            current_time = time.time()
            
            if phase == 'counting':
                # Counting phase - detect people in real-time
                elapsed_time = current_time - start_time
                time_remaining = self.detection_time - elapsed_time
                
                if time_remaining > 0:
                    # Run detection on EVERY frame for real-time counting
                    results = self.model(frame, verbose=False, classes=[0], conf=0.5)
                    
                    # Count people CURRENTLY in the zone
                    current_count = 0
                    
                    if results[0].boxes is not None and len(results[0].boxes) > 0:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        classes = results[0].boxes.cls.cpu().numpy()
                        confidences = results[0].boxes.conf.cpu().numpy()
                        
                        # Count and draw each person in the zone
                        for box, cls, conf in zip(boxes, classes, confidences):
                            if int(cls) == 0:  # Person class
                                if self.is_in_boundary(box):
                                    current_count += 1
                                    
                                    # Draw detection box
                                    x1, y1, x2, y2 = map(int, box)
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(frame, f"Person #{current_count}", (x1, y1 - 10),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Store the current count (this is what we'll use for decision)
                    final_count = current_count
                    
                    # Draw UI
                    frame = self.draw_ui(frame, time_remaining, current_count, phase)
                    
                else:
                    # Counting phase ended - make decision based on FINAL count
                    if final_count == 0:
                        # NO PEDESTRIANS: Keep vehicle green
                        print("\n" + "="*60)
                        print("DETECTION COMPLETED - NO PEDESTRIANS IN ZONE")
                        print("="*60)
                        print("Pedestrian Light: RED")
                        print("Vehicle Light: GREEN")
                        print("Restarting detection in 3 seconds...\n")
                        
                        phase = 'vehicle_green'
                        start_time = time.time()
                        
                    else:
                        # PEDESTRIANS DETECTED: Switch lights
                        green_light_duration = self.calculate_green_light_time(final_count)
                        
                        print("\n" + "="*60)
                        print("DETECTION COMPLETED - PEDESTRIANS IN ZONE")
                        print("="*60)
                        print(f"People in zone: {final_count}")
                        print(f"Pedestrian Green Light: {green_light_duration} seconds")
                        print("Pedestrian Light: GREEN")
                        print("Vehicle Light: RED")
                        print("="*60 + "\n")
                        
                        # Switch lights
                        self.pedestrian_light = "GREEN"
                        self.vehicle_light = "RED"
                        
                        phase = 'pedestrian_green'
                        start_time = time.time()
                        greenlight_duration = green_light_duration
            
            elif phase == 'pedestrian_green':
                # Pedestrian green light phase
                elapsed_time = current_time - start_time
                time_remaining = greenlight_duration - elapsed_time
                
                if time_remaining > 0:
                    frame = self.draw_ui(frame, time_remaining, final_count, phase)
                else:
                    # Pedestrian green ended
                    print("\n" + "="*60)
                    print("PEDESTRIAN CROSSING COMPLETED")
                    print("="*60)
                    print("Switching back to vehicle flow...")
                    print("Pedestrian Light: RED")
                    print("Vehicle Light: GREEN")
                    print("="*60 + "\n")
                    
                    self.pedestrian_light = "RED"
                    self.vehicle_light = "GREEN"
                    
                    # Restart counting
                    phase = 'counting'
                    start_time = time.time()
                    final_count = 0
            
            elif phase == 'vehicle_green':
                # Vehicle green (no pedestrians)
                elapsed_time = current_time - start_time
                
                if elapsed_time > 3:  # Show for 3 seconds
                    # Restart counting
                    phase = 'counting'
                    start_time = time.time()
                    final_count = 0
                else:
                    frame = self.draw_ui(frame, 0, 0, phase)
            
            # Display frame
            cv2.imshow('Real-Time Traffic Light System - Camera Feed', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\nSystem stopped by user")
                break
            elif key == ord('r') or key == ord('R'):
                print("\nRestarting system...")
                self.pedestrian_light = "RED"
                self.vehicle_light = "GREEN"
                phase = 'counting'
                start_time = time.time()
                final_count = 0
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\nSystem terminated successfully")


def main():
    """Main function to run the system"""
    
    print("="*60)
    print("REAL-TIME TRAFFIC LIGHT SYSTEM")
    print("Using YOLOv11 with COCO Pre-trained Model")
    print("Camera-Based Detection System")
    print("="*60)
    
    # CONFIGURATION - Adjust these values
    DETECTION_TIME = 20  # Time in seconds before making decision
    CAMERA_INDEX = 1     # 0 = main camera, 1 = secondary camera, etc.
    
    print(f"\nConfiguration:")
    print(f"  Detection Time: {DETECTION_TIME} seconds")
    print(f"  Camera Device: {CAMERA_INDEX}")
    print("\nStarting camera system...\n")
    
    # Create and run system
    system = RealTimeTrafficLightSystem(
        detection_time=DETECTION_TIME,
        camera_index=CAMERA_INDEX
    )
    
    system.run()


if __name__ == "__main__":
    main()