from PyQt5.QtWidgets import QApplication, QMessageBox, QDialog, QFormLayout, QLineEdit, QPushButton, QLabel
from PyQt5.QtCore import Qt
import cv2
import face_recognition
import sqlite3
import numpy as np
import sys
import time
import json

db_path = 'landMarks.db'

def initialize_detector():
    model_path = "face_detection_yunet_2023mar.onnx"
    detector = cv2.FaceDetectorYN.create(model_path, "", (640, 640))
    return detector

def detect_faces(detector, image):
    # Yüz tespiti
    _, faces = detector.detect(image)
    return faces

def encode_faces(image):
    encodings = face_recognition.face_encodings(image)
    if encodings:
        return encodings[0]
    return None

def save_encoding_and_landmarks_to_db(user_id, encoding, landmarks):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS face_data
                          (user_id TEXT, encoding BLOB, chin TEXT, left_eyebrow TEXT, right_eyebrow TEXT, 
                           nose_bridge TEXT, nose_tip TEXT, left_eye TEXT, right_eye TEXT, 
                           top_lip TEXT, bottom_lip TEXT)''')
        cursor.execute('''INSERT INTO face_data 
                          (user_id, encoding, chin, left_eyebrow, right_eyebrow, nose_bridge, nose_tip, 
                           left_eye, right_eye, top_lip, bottom_lip) 
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                       (user_id,
                        encoding.tobytes(),
                        json.dumps(landmarks.get('chin', [])),
                        json.dumps(landmarks.get('left_eyebrow', [])),
                        json.dumps(landmarks.get('right_eyebrow', [])),
                        json.dumps(landmarks.get('nose_bridge', [])),
                        json.dumps(landmarks.get('nose_tip', [])),
                        json.dumps(landmarks.get('left_eye', [])),
                        json.dumps(landmarks.get('right_eye', [])),
                        json.dumps(landmarks.get('top_lip', [])),
                        json.dumps(landmarks.get('bottom_lip', []))))
        print("Veriler veri tabanına kaydedildi!")
        conn.commit()

class LoginPage(QDialog):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.setFixedSize(250, 250)
    
    def init_ui(self):
        self.setWindowTitle("Login Page")
        layout = QFormLayout()

        self.userName = QLineEdit()
        layout.addRow(QLabel("User Name : "), self.userName)

        loginBtn = QPushButton("Login!")
        layout.addWidget(loginBtn)
        loginBtn.clicked.connect(self.login)

        registerBtn = QPushButton("Register to me!")
        layout.addWidget(registerBtn)
        registerBtn.clicked.connect(self.Register)

        self.setLayout(layout)

    def login(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        prev_frame_time = 0
        new_frame_time = 0
        
        cv2.namedWindow("Giris Yap")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            font = cv2.FONT_HERSHEY_SIMPLEX
            new_frame_time = time.time()

            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            fps = int(fps)
            fps = str(fps)

            face_locations = face_recognition.face_locations(frame, model="hog")
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            face_landmarks = face_recognition.face_landmarks(frame)
            if face_encodings:
                for (face_location, encoding, landmarks) in zip(face_locations, face_encodings, face_landmarks):
                    name = self.get_face_data_from_db(encoding, landmarks)
                    if name:
                        print(f"Giris Basarili Kullanici {name}")
                        top, right, bottom, left = face_location
                        cv2.putText(frame, name, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    else:
                        print("Kullanici Bulunamadi")
            else:
                print("Yuz bulunamadi")

            cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow("Giris Yap", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    # Koordinatların mesafesini hesaplama fonksiyonu
    def calculate_chin_distance(self,chin1, chin2):
        chin1 = np.array(chin1)
        chin2 = np.array(chin2)
        # Koordinatların aynı sıraya sahip olduğunu varsayar
        return np.sqrt(np.sum((chin1 - chin2) ** 2))
    # def calculate_eyebrow_distance(self,eyebrow1, eyebrow2):
    #     eyebrow1 = np.array(eyebrow1)
    #     eyebrow2 = np.array(eyebrow2)
    #     # Koordinatların aynı sıraya sahip olduğunu varsayar
        return np.sqrt(np.sum((eyebrow1 - eyebrow2) ** 2))
    def calculate_left_eye_distance(self,eyeleft1,eyeleft2):
        eyeleft1 = np.array(eyeleft1)
        eyeleft2 = np.array(eyeleft2)
        # Koordinatların aynı sıraya sahip olduğunu varsayar
        return np.sqrt(np.sum((eyeleft1 - eyeleft2) ** 2))
    def calculate_right_eye_distance(self,eyeright1, eyeright2):
        eyeright1 = np.array(eyeright1)
        eyeright2 = np.array(eyeright2)
        # Koordinatların aynı sıraya sahip olduğunu varsayar
        return np.sqrt(np.sum((eyeright1 - eyeright2) ** 2))

    # Veritabanından chin verisini JSON formatında çözme
    def parse_chin_data(self,chin_blob):
        return json.loads(chin_blob)
    def parse_left_eye(self,eye_left):
        return json.loads(eye_left)
    def parse_right_eye(self,eye_right):
        return json.loads(eye_right)

    # Veritabanından yüz verilerini alma ve karşılaştırma
    def get_face_data_from_db(self,encoding, landmarks):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_id, encoding, chin, left_eyebrow, right_eyebrow, nose_bridge, nose_tip, left_eye, right_eye, top_lip, bottom_lip
            FROM face_data
        ''')
        rows = cursor.fetchall()
        conn.close()

        chin_threshold = 0.6  # Chin mesafesi için eşik değeri
        left_eye_threshold = 0.4  # Sol göz mesafesi için eşik değeri
        right_eye_threshold = 0.4  # Sağ göz mesafesi için eşik değeri

        best_match_user_id = None
        min_chin_distance = float('inf')
        min_left_eye_distance = float('inf')
        min_right_eye_distance = float('inf')

        for row in rows:
            user_id = row[0]
            encoding_blob = row[1]
            chin_blob = row[2]
            leftEyeBlob = row[7]
            rightEyeBlob = row[8]

            known_encoding = np.frombuffer(encoding_blob, dtype=np.float64)

            encoding_match = face_recognition.compare_faces([known_encoding], encoding, tolerance=0.55)
            chin_coordinates = self.parse_chin_data(chin_blob)
            left_eye_coordinates = self.parse_left_eye(leftEyeBlob)
            right_eye_coordinates = self.parse_right_eye(rightEyeBlob)
            print(f"LANDMARKS BOS MUUUUUUUUU : {landmarks}")
            if landmarks:
                print("LANDMARKS GİRDİ")
                for face_landmarks in landmarks:
                    if 'chin' in face_landmarks and 'left_eye' in face_landmarks and 'right_eye' in face_landmarks:
                        print("ıf e GİRDİİİİİİ")
                        chin_distance = self.calculate_chin_distance(chin_coordinates, face_landmarks['chin'])
                        left_eye_distance = self.calculate_left_eye_distance(left_eye_coordinates, face_landmarks['left_eye'])
                        right_eye_distance = self.calculate_right_eye_distance(right_eye_coordinates, face_landmarks['right_eye'])
                        print(f"Chin Distance : {chin_distance}")
                        print(f"Left Eye Distance : {left_eye_distance}")
                        print(f"Right Eye Distance : {right_eye_distance}")

                        if encoding_match[0]:
                            if chin_distance < chin_threshold and left_eye_distance < left_eye_threshold and right_eye_distance < right_eye_threshold:
                                best_match_user_id = user_id
                                break

        return best_match_user_id
                

    def Register(self):
        register = Register()
        register.exec_()

class Register(QDialog):
    def __init__(self):
        super().__init__()
        self.registerForm()
        self.setFixedSize(250, 250)

    def registerForm(self):
        layout = QFormLayout()

        self.userName = QLineEdit()
        layout.addRow(QLabel("User Name : "), self.userName)

        captureBtn = QPushButton("Capture and Save")
        layout.addWidget(captureBtn)
        captureBtn.clicked.connect(self.capture_photos)

        self.setLayout(layout)

    def capture_photos(self):
        username = self.userName.text()
        if not username:
            QMessageBox.warning(self, "Error", "UserName is required!")
            return

        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not camera.isOpened():
            QMessageBox.warning(self, "Error", "Kamera Kullanılamıyor")
            return
        photo_paths = ["photo_front.jpg", "photo_right.jpg"]
        angles = ["ön", "sağ"]
        for path, angle in zip(photo_paths, angles):
            print(f"{angle} açıdan fotoğraf çekiliyor...")
            while True:
                ret, frame = camera.read()
                if ret:
                    cv2.imshow('Kamera', frame)
                    if cv2.waitKey(1) & 0xFF == ord('c'):
                        frame_resized = cv2.resize(frame, (640, 640))
                        cv2.imwrite(path, frame_resized)
                        image = cv2.imread(path)
                        encoding = encode_faces(image)
                        landmarks = face_recognition.face_landmarks(image)
                        if encoding is not None and landmarks:
                            QMessageBox.information(self,"Bilgi","Fotograf Encodlandi ve veri tabanına kaydedildi")
                            save_encoding_and_landmarks_to_db(username,encoding,landmarks[0])
                            break
                        else:
                            QMessageBox.warning(self,"Error","Fotograf Encodlanamadi")
                    # image = cv2.imread(path)
                        # encoding = encode_faces(image)
                        # if encoding is not None:
                        #     print("Encoding başarılı")
                        #     print("Veri tabanına kaydedildi")
                        #     save_encoding_to_db(username, encoding)
                        # else:
                        #     print("Fotograf Encodlanamadı")
            cv2.destroyWindow('Kamera')
            print(f"{angle} açıdan fotoğraf çekildi.")

            # image = cv2.imread(path)
            # encoding = encode_faces(image)
            # if encoding is not None:
            #     print("Encoding başarılı")
            #     print("Veri tabanına kaydedildi")
            #     save_encoding_to_db(username, encoding)
            # else:
            #     print("Fotograf Encodlanamadı")

        camera.release()
        cv2.destroyAllWindows()

        # Fotoğrafları işleyip veritabanına kaydet
        # for path in photo_paths:
        #     print(f"Processing {path}...")
        #     image = cv2.imread(path)
        #     print(f"İmage Size : {image.size}")
        #     encoding = encode_faces(image)
        #     if encoding is not None:
        #         print("Encoding başarılı")
        #         print("Veri tabanına kaydedildi")
        #         save_encoding_to_db(username, encoding)
        #     else:
        #         print("Fotograf Encodlanamadı")


            # if image is not None:
            #     faces = detect_faces(detector, image)
            #     if faces is not None:
            #         print(f"Faces detected: {faces}")
            #         for face in faces:
            #             x1, y1, x2, y2 = map(int, face[:4])
            #             face_image = image[y1:y2, x1:x2]
            #             print(f"Face Image Size : {face_image.size}")
            #             if face_image.size > 0 : 
            #                 face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            #                 encoding = encode_faces(face_image_rgb)
            #                 print(f"Encoding {encoding}")
            #                 if encoding is not None:
            #                     print("Encoding obtained.")
            #                     print("Saving to database...")
            #                     save_encoding_to_db(username, encoding)
            #                 else:
            #                     print("No encoding found.")
            #             else:
            #                 print("FACE IMAGE EMPTY")
            #     else:
            #         print("No faces detected.")
            # else:
            #     print(f"Failed to load image from {path}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    login_page = LoginPage()
    login_page.show()
    sys.exit(app.exec_())
