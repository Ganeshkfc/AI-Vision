import os
import cv2
import time
import threading
import numpy as np
from datetime import datetime

# --- KIVY UI COMPONENTS ---
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.clock import Clock
from kivy.utils import platform

try:
    from jnius import autoclass
except ImportError:
    autoclass = None

# TFLite for Mobile AI
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        from tensorflow import lite as tflite
    except ImportError:
        tflite = None

class VisionApp(App):
    def build(self):
        # --- YOUR ORIGINAL SETTINGS (KEEPING LOGIC) ---
        self.current_mode = 1 
        self.KNOWN_WIDTHS = {'person': 50, 'chair': 45, 'bottle': 8, 'cell phone': 7}
        self.FOCAL_LENGTH = 715 
        self.METRIC_THRESHOLD_CM = 91.44
        self.last_speech_time = 0
        self.SPEECH_COOLDOWN = 6 

        # YOLOv8 Class Names (Standard COCO)
        self.classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        # Setup TFLite
        self.interpreter = tflite.Interpreter(model_path="yolov8n_float32.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Setup TTS for Android
        self.tts = None
        if platform == 'android' and autoclass:
            try:
                PythonActivity = autoclass('org.kivy.android.PythonActivity')
                self.tts = autoclass('android.speech.tts.TextToSpeech')(PythonActivity.mActivity, None)
            except: pass

        # --- GESTURE BASED UI (KEEPING YOUR DESIGN) ---
        layout = BoxLayout(orientation='vertical')

        # Mobile Camera Widget
        self.cam = Camera(play=True, resolution=(640, 480), index=0)
        layout.add_widget(self.cam)

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
        else:
            print(f"Speech: {text}")

    def get_distance_cm(self, label, width_px):
        real_w = self.KNOWN_WIDTHS.get(label, 30)
        return (real_w * self.FOCAL_LENGTH) / width_px

    def ai_engine(self):
        while True:
            if not self.cam.texture:
                time.sleep(0.2)
                continue

            # 1. Capture Frame from Kivy Camera
            texture = self.cam.texture
            frame = np.frombuffer(texture.pixels, dtype='uint8')
            frame = frame.reshape(texture.height, texture.width, 4)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            f_h, f_w, _ = frame.shape

            # 2. Prepare for TFLite
            input_data = cv2.resize(frame, (640, 640))
            input_data = input_data.astype(np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=0)

            # 3. Inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            output = output.transpose() # New YOLO format: [8400, 84]

            # 4. Filter Detections
            now = time.time()
            if now - self.last_speech_time > self.SPEECH_COOLDOWN:
                detections = []
                for row in output:
                    prob = row[4:].max()
                    if prob > 0.4:
                        class_id = row[4:].argmax()
                        label = self.classes[class_id]
                        xc, yc, w, h = row[:4]
                        x1 = (xc - w/2) * (f_w / 640)
                        x2 = (xc + w/2) * (f_w / 640)
                        detections.append({'label': label, 'x1': x1, 'x2': x2, 'prob': prob})

                if detections:
                    if self.current_mode == 1:
                        # --- MODE 1 LOGIC ---
                        items = []
                        seen = set()
                        for d in detections:
                            if d['label'] not in seen:
                                x_center = (d['x1'] + d['x2']) / 2
                                dir_s = "on your left" if x_center < (f_w/3) else "in front of you" if x_center < (2*f_w/3) else "on your right"
                                items.append(f"a {d['label']} {dir_s}")
                                seen.add(d['label'])
                        self.speak("I see " + " and ".join(items))
                    else:
                        # --- MODE 2 LOGIC ---
                        best = max(detections, key=lambda x: x['prob'])
                        x_center = (best['x1'] + best['x2']) / 2
                        dir_s = "on your left" if x_center < (f_w/3) else "in front of you" if x_center < (2*f_w/3) else "on your right"
                        d_cm = self.get_distance_cm(best['label'], best['x2'] - best['x1'])
                        d_str = f"{int(d_cm)} centimeters" if d_cm < self.METRIC_THRESHOLD_CM else f"{round(d_cm/30.48, 1)} feet"
                        self.speak(f"I see a {best['label']}, {d_str}, {dir_s}")
                    
                    self.last_speech_time = now
            
            time.sleep(0.3)

if __name__ == "__main__":
    VisionApp().run()
