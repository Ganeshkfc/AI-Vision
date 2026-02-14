import os
import cv2
import time
import threading
from datetime import datetime
from ultralytics import YOLO

# --- KIVY UI COMPONENTS ---
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from jnius import autoclass

class VisionApp(App):
    def build(self):
        # Settings
        self.current_mode = 1 
        self.KNOWN_WIDTHS = {'person': 50, 'chair': 45, 'bottle': 8, 'cell phone': 7}
        self.FOCAL_LENGTH = 715 
        
        # Threshold: 91.44 cm (Equivalent to 3 feet)
        self.METRIC_THRESHOLD_CM = 91.44

        # Setup TTS
        try:
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            self.tts = autoclass('android.speech.tts.TextToSpeech')(PythonActivity.mActivity, None)
        except: self.tts = None

        self.model = YOLO('yolov8n.pt')
        self.last_speech_time = 0
        self.SPEECH_COOLDOWN = 6 

        # --- GESTURE BASED UI ---
        layout = BoxLayout(orientation='vertical')

        self.top_btn = Button(
            text="TAP TOP HALF TO CHANGE MODE\n(Mode 1: Multi-Object Active)",
            background_color=(0.1, 0.5, 0.8, 1),
            font_size='20sp',
            halign='center'
        )
        self.top_btn.bind(on_release=self.toggle_mode)

        self.bottom_btn = Button(
            text="TAP BOTTOM HALF TO CLOSE APP",
            background_color=(0.8, 0.2, 0.2, 1),
            font_size='20sp',
            halign='center'
        )
        self.bottom_btn.bind(on_release=self.check_close_app)

        layout.add_widget(self.top_btn)
        layout.add_widget(self.bottom_btn)

        Clock.schedule_once(lambda dt: self.speak("System ready. Mode 1 active. Detecting multiple objects."), 2)
        
        threading.Thread(target=self.ai_engine, daemon=True).start()
        return layout

    def toggle_mode(self, instance):
        if self.current_mode == 1:
            self.current_mode = 2
            self.speak("Mode 2 activated. Precision distance detection in feet enabled.")
            self.top_btn.text = "MODE 2 ACTIVE\n(Single Object + Distance in Feet)"
        else:
            self.current_mode = 1
            self.speak("Mode 1 activated. Multi-object detection enabled.")
            self.top_btn.text = "MODE 1 ACTIVE\n(Multiple Objects)"

    def check_close_app(self, instance):
        self.speak("Closing application. Goodbye.")
        Clock.schedule_once(lambda dt: self.stop(), 1)

    def speak(self, text):
        if self.tts:
            self.tts.setSpeechRate(0.8) 
            self.tts.speak(text, 0, None) 

    def get_distance_cm(self, label, width_px):
        real_w = self.KNOWN_WIDTHS.get(label, 30)
        return (real_w * self.FOCAL_LENGTH) / width_px

    def ai_engine(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret: continue
            f_h, f_w, _ = frame.shape
            results = self.model(frame, conf=0.4, verbose=False)
            
            if results and len(results[0].boxes) > 0:
                now = time.time()
                if now - self.last_speech_time > self.SPEECH_COOLDOWN:
                    
                    if self.current_mode == 1:
                        # --- MODE 1: MULTI-OBJECT DETECTION ---
                        detected_items = []
                        seen_labels = set()
                        
                        for box in results[0].boxes:
                            label = self.model.names[int(box.cls[0])]
                            if label not in seen_labels:
                                x1, _, x2, _ = box.xyxy[0].tolist()
                                x_center = (x1 + x2) / 2
                                
                                if x_center < (f_w / 3): dir_s = "on your left"
                                elif x_center < (2 * f_w / 3): dir_s = "in front of you"
                                else: dir_s = "on your right"
                                
                                detected_items.append(f"a {label} {dir_s}")
                                seen_labels.add(label)
                        
                        if detected_items:
                            announcement = "I see " + " and ".join(detected_items)
                            self.speak(announcement)
                            self.last_speech_time = now

                    else:
                        # --- MODE 2: SINGLE-OBJECT PRECISION (CM & FEET) ---
                        best_box = None
                        min_dist = float('inf')
                        for box in results[0].boxes:
                            x1, _, x2, _ = box.xyxy[0].tolist()
                            c_dist = abs(((x1 + x2) / 2) - (f_w / 2))
                            if c_dist < min_dist:
                                min_dist = c_dist
                                best_box = box

                        label = self.model.names[int(best_box.cls[0])]
                        x1, _, x2, _ = best_box.xyxy[0].tolist()
                        x_center = (x1 + x2) / 2
                        
                        if x_center < (f_w / 3): dir_s = "on your left"
                        elif x_center < (2 * f_w / 3): dir_s = "in front of you"
                        else: dir_s = "on your right"

                        d_cm = self.get_distance_cm(label, x2 - x1)
                        
                        # UPDATED UNIT CONVERSION: CM AND FEET
                        if d_cm < self.METRIC_THRESHOLD_CM:
                            d_str = f"{int(d_cm)} centimeters"
                        else:
                            # Convert CM to Feet (Divide by 30.48)
                            d_feet = round(d_cm / 30.48, 1)
                            d_str = f"{d_feet} feet"
                        
                        self.speak(f"I see a {label}, {d_str}, {dir_s}")
                        self.last_speech_time = now
            
            time.sleep(0.1)

if __name__ == "__main__":
    VisionApp().run()
