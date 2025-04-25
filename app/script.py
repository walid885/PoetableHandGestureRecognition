import cv2
import mediapipe as mp
import numpy as np
import time

class HandGestureRecognizer:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Gesture definitions
        self.gestures = {
            "open_palm": self._is_open_palm,
            "closed_fist": self._is_closed_fist,
            "pointing": self._is_pointing,
            "victory": self._is_victory,
            "thumbs_up": self._is_thumbs_up
        }
        
        # Debug mode flag
        self.debug = True
        
    def _is_open_palm(self, landmarks):
        # All fingers extended
        finger_tips = [self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                      self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                      self.mp_hands.HandLandmark.RING_FINGER_TIP,
                      self.mp_hands.HandLandmark.PINKY_TIP]
        
        finger_mcp = [self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
                     self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                     self.mp_hands.HandLandmark.RING_FINGER_MCP,
                     self.mp_hands.HandLandmark.PINKY_MCP]
        
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        
        # Check if all fingertips are higher than their MCP joints
        fingers_extended = True
        for tip, mcp in zip(finger_tips, finger_mcp):
            if landmarks[tip].y > landmarks[mcp].y:
                fingers_extended = False
                break
                
        # Check thumb position separately
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_mcp = landmarks[self.mp_hands.HandLandmark.THUMB_MCP]
        
        if landmarks[self.mp_hands.HandLandmark.THUMB_TIP].x < landmarks[self.mp_hands.HandLandmark.THUMB_MCP].x:
            thumb_extended = True
        else:
            thumb_extended = False
            
        return fingers_extended and thumb_extended
    
    def _is_closed_fist(self, landmarks):
        finger_tips = [self.mp_hands.HandLandmark.THUMB_TIP,
                      self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                      self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                      self.mp_hands.HandLandmark.RING_FINGER_TIP,
                      self.mp_hands.HandLandmark.PINKY_TIP]
        
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        
        # For a fist, all fingertips should be close to the palm
        palm_center = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        
        # Check if all fingers are curled (y-coordinate below their MCP joint)
        fingers_curled = True
        for i, tip in enumerate(finger_tips):
            if i == 0:  # Thumb
                continue  # Skip thumb for simplicity
            
            tip_pos = landmarks[tip]
            mcp_pos = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_MCP + (i-1)*4]
            
            if tip_pos.y < mcp_pos.y:  # If tip is above MCP joint
                fingers_curled = False
                break
                
        return fingers_curled
    
    def _is_pointing(self, landmarks):
        # Index finger extended, other fingers curled
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        
        # Check that index finger is extended
        index_extended = index_tip.y < index_pip.y
        
        # Check that other fingers are curled
        other_fingers_curled = True
        
        # Middle finger
        if landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y:
            other_fingers_curled = False
            
        # Ring finger
        if landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP].y:
            other_fingers_curled = False
            
        # Pinky
        if landmarks[self.mp_hands.HandLandmark.PINKY_TIP].y < landmarks[self.mp_hands.HandLandmark.PINKY_PIP].y:
            other_fingers_curled = False
            
        return index_extended and other_fingers_curled
    
    def _is_victory(self, landmarks):
        # Index and middle fingers extended, others curled
        index_extended = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        middle_extended = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        
        # Check that other fingers are curled
        ring_curled = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP].y > landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP].y
        pinky_curled = landmarks[self.mp_hands.HandLandmark.PINKY_TIP].y > landmarks[self.mp_hands.HandLandmark.PINKY_PIP].y
        
        return index_extended and middle_extended and ring_curled and pinky_curled
    
    def _is_thumbs_up(self, landmarks):
        # Thumb extended upward, other fingers curled
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_mcp = landmarks[self.mp_hands.HandLandmark.THUMB_MCP]
        wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
        
        # Check if thumb is pointing upward
        thumb_up = thumb_tip.y < thumb_mcp.y
        
        # Check if other fingers are curled
        other_fingers_curled = True
        
        finger_tips = [self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                      self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                      self.mp_hands.HandLandmark.RING_FINGER_TIP,
                      self.mp_hands.HandLandmark.PINKY_TIP]
        
        finger_pips = [self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
                      self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                      self.mp_hands.HandLandmark.RING_FINGER_PIP,
                      self.mp_hands.HandLandmark.PINKY_PIP]
        
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip].y < landmarks[pip].y:
                other_fingers_curled = False
                break
                
        return thumb_up and other_fingers_curled
        
    def recognize_gesture(self, hand_landmarks):
        if hand_landmarks is None:
            return "No gesture"
            
        # Convert landmarks to normalized format
        landmarks = {}
        for i, landmark in enumerate(hand_landmarks.landmark):
            landmarks[i] = landmark
            
        detected_gestures = []
        for gesture_name, gesture_func in self.gestures.items():
            if gesture_func(landmarks):
                detected_gestures.append(gesture_name)
                
        if not detected_gestures:
            return "Unknown gesture"
        else:
            return detected_gestures[0]  # Return first detected gesture
        
    def process_frame(self, frame):
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Recognize gesture for this hand
                gesture = self.recognize_gesture(hand_landmarks)
                
                # Display recognized gesture
                cv2.putText(frame, f"Gesture: {gesture}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
        return frame, results.multi_hand_landmarks

def main():
    print("Starting Hand Gesture Recognition...")
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    recognizer = HandGestureRecognizer()
    
    print("Camera initialized. Press 'q' to quit.")
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
            
        # Process frame for hand gestures
        processed_frame, _ = recognizer.process_frame(frame)
        
        # Display the processed frame
        cv2.imshow('MediaPipe Hand Gesture Recognition', processed_frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed successfully.")

if __name__ == "__main__":
    main()