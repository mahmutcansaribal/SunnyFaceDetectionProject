from PyQt5.QtWidgets import QApplication, QMessageBox, QDialog, QFormLayout, QLineEdit, QPushButton, QLabel
from PyQt5.QtCore import Qt
import cv2
import face_recognition
import sqlite3
import numpy as np
import sys
import time
import json
import os

db_path = 'landMarks.db'
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
        
        cv2.namedWindow("Kamera")
        
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

            face_locations = face_recognition.face_locations(frame,model="hog")
            face_landmarks = face_recognition.face_landmarks(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            print(f"face Locations : {face_locations}")
            for (faceler, encoding, landmarks) in zip(face_locations, face_encodings, face_landmarks):
                print(f"FACELER {faceler}")
                name = self.get_face_data_from_db(encoding, landmarks)
                top,right,bottom,left = faceler
                cv2.putText(frame, name, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("Kamera",frame)

            '''
            
            YAPILAN DEGİSİKLER

            HER BİR YÜZ LOCASYONU İÇİN ENCODING VE LANDMARKS DEGERLERİ FOR ICERİSİNE ALINARAK VERİ TABANINA GÖNDERİLDİ
            YORUM SATIRI İLE YUKARI KOD PARÇASI FARKLARI GÖRÜNMEKTEDİR.
            '''

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
    def calculate_top_lib_distance(self,toplip1,toplip2):
        toplip1 = np.array(toplip1)
        toplip2 = np.array(toplip2)
        return np.sqrt(np.sum((toplip1-toplip2)))
    def calculate_bottom_lib_distance(self,bottomlip1,bottomlip2):
        bottomlip1 = np.array(bottomlip1)
        bottomlip2 = np.array(bottomlip2)
        return np.sqrt(np.sum((bottomlip1-bottomlip2)))

    # Veritabanından chin verisini JSON formatında çözme
    def parse_chin_data(self,chin_blob):
        return json.loads(chin_blob)
    def parse_left_eye(self,eye_left):
        return json.loads(eye_left)
    def parse_right_eye(self,eye_right):
        return json.loads(eye_right)
    def parse_top_lib(self,top_lib):
        return json.loads(top_lib)
    def parse_bottom_lib(self,bottom_lib):
        return json.loads(bottom_lib)

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

        best_match_user_id = None
        min_chin_distance = float('inf')
        min_left_eye_distance = float('inf')
        min_right_eye_distance = float('inf')
        min_top_lip_distance = float('inf')
        min_bottom_lip_distance = float('inf')

        for row in rows:
            user_id = row[0]
            encoding_blob = row[1]
            chin_blob = row[2]
            leftEyeBlob = row[7]
            rightEyeBlob = row[8]
            topLibBlob = row[9]
            bottomLibBlob = row[10]

            known_encoding = np.frombuffer(encoding_blob, dtype=np.float64)

            encoding_match = face_recognition.compare_faces([known_encoding], encoding, tolerance=0.57)
            
            chin_coordinates = self.parse_chin_data(chin_blob)
            left_eye_coordinates = self.parse_left_eye(leftEyeBlob)
            right_eye_coordinates = self.parse_right_eye(rightEyeBlob)
            top_lib_coordinates = self.parse_top_lib(topLibBlob)
            bottom_lib_coordinates = self.parse_bottom_lib(bottomLibBlob)

            # if landmarks and 'chin'  and 'left_eye' and 'right_eye' in landmarks[0] :
            '''
            YAPILAN DEGİSİKLİKLER 
            1- HER BİR LANDMARKS VE ENCODING DEGERİ  İÇİN LANDMARKS[0] KALDIRILDI
            2- IF YAPISI DÜZENLENDİ
            '''
        

            if 'chin' in landmarks and 'left_eye' in landmarks and 'right_eye' in landmarks and 'top_lip' in landmarks and 'bottom_lip' in landmarks :
                chin_distance = self.calculate_chin_distance(chin_coordinates, landmarks['chin'])
                left_eye_distance = self.calculate_left_eye_distance(left_eye_coordinates,landmarks['left_eye'])
                right_eye_distance = self.calculate_right_eye_distance(right_eye_coordinates,landmarks['right_eye'])
                top_lib_distance = self.calculate_top_lib_distance(top_lib_coordinates,landmarks['top_lip'])
                bottom_lib_distance = self.calculate_bottom_lib_distance(bottom_lib_coordinates,landmarks['bottom_lip'])
                print(f"Chin Distance : {chin_distance}")
                print(f"Left Eye Distance : {left_eye_distance}")
                print(f"Right Eye Distance : {right_eye_distance}")
                print(f"top lib Distance : {top_lib_distance}")
                print(f"bottom lib Distance : {bottom_lib_distance}")

                if encoding_match[0] and chin_distance < min_chin_distance and left_eye_distance < min_left_eye_distance and right_eye_distance < min_right_eye_distance and top_lib_distance < min_top_lip_distance and bottom_lib_distance < min_bottom_lip_distance:
                    best_match_user_id = user_id
                    min_chin_distance = chin_distance
                    min_left_eye_distance = left_eye_distance
                    min_right_eye_distance = right_eye_distance
                    min_top_lip_distance = top_lib_distance
                    min_bottom_lip_distance = bottom_lib_distance
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
                            os.remove(path)
                            break
                        else:
                            QMessageBox.warning(self,"Error","Fotograf Encodlanamadi")
            cv2.destroyWindow('Kamera')
            print(f"{angle} açıdan fotoğraf çekildi.")
        camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    login_page = LoginPage()
    login_page.show()
    sys.exit(app.exec_())
