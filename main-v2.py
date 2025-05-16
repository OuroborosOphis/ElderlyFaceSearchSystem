import os
import cv2
import numpy as np
import sqlite3
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

class ElderlyFaceSystem:
    def __init__(self, db_path="elderly_faces.db"):
        self.db_path = db_path
        self.image_size = (128, 128)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Khởi tạo cơ sở dữ liệu
        self.init_database()
        self.load_existing_data()

    def init_database(self):
        """Khởi tạo cấu trúc cơ sở dữ liệu"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS faces (
                        id INTEGER PRIMARY KEY,
                        path TEXT UNIQUE,
                        features BLOB,
                        age INTEGER,
                        gender TEXT)''')
        conn.commit()
        conn.close()

    def load_existing_data(self):
        """Tải dữ liệu đã có từ database"""
        self.features = []
        self.metadata = []
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT path, features, age, gender FROM faces")
        for row in cursor.fetchall():
            self.metadata.append({
                "path": row[0],
                "age": row[2],
                "gender": row[3]
            })
            self.features.append(np.frombuffer(row[1], dtype=np.float32))
        conn.close()

    def preprocess_image(self, image):
        """Tiền xử lý ảnh và phát hiện khuôn mặt"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return cv2.resize(image, self.image_size)
        
        (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
        face_roi = image[y:y+h, x:x+w]
        return cv2.resize(face_roi, self.image_size)

    def extract_features(self, image):
        """Trích xuất đặc trưng tổng hợp"""
        return {
            "lbp": self.lbp_features(image),
            "geometry": self.geometric_features(image),
            "color": self.color_features(image)
        }

    def lbp_features(self, image, radius=1, neighbors=8):
        """Triển khai LBP thủ công"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        lbp = np.zeros_like(gray)
        
        for y in range(radius, height-radius):
            for x in range(radius, width-radius):
                center = gray[y, x]
                code = 0
                for i in range(neighbors):
                    angle = 2 * np.pi * i / neighbors
                    ny = y + int(radius * np.sin(angle))
                    nx = x + int(radius * np.cos(angle))
                    code |= (gray[ny, nx] >= center) << (neighbors-1-i)
                lbp[y, x] = code
        
        hist, _ = np.histogram(lbp, bins=59, range=(0, 59))
        return hist / hist.sum()

    def geometric_features(self, image):
        """Tính đặc trưng hình học"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if not faces:
            return np.zeros(3)
        
        x, y, w, h = faces[0]
        return np.array([h/w, (y+h)/self.image_size[0], w/self.image_size[1]])

    def color_features(self, image):
        """Phân tích màu sắc"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Phát hiện vùng da
        lower_skin = np.array([0, 48, 80], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_area = cv2.mean(hsv, mask=skin_mask)[:3]
        
        # Phát hiện tóc bạc
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        white_hair = cv2.mean(white_mask)[0]/255
        
        return np.concatenate([skin_area, [white_hair]])

    def add_to_database(self, image_path, age=None, gender=None):
        """Thêm ảnh mới vào database"""
        image = cv2.imread(image_path)
        if image is None:
            return False
        
        processed = self.preprocess_image(image)
        features = self.extract_features(processed)
        
        # Chuẩn bị dữ liệu để lưu
        feature_vector = np.concatenate([
            features["lbp"],
            features["geometry"],
            features["color"]
        ]).astype(np.float32)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO faces (path, features, age, gender)
                        VALUES (?, ?, ?, ?)''',
                     (image_path, feature_vector.tobytes(), age, gender))
        conn.commit()
        conn.close()
        
        # Cập nhật bộ nhớ
        self.features.append(feature_vector)
        self.metadata.append({"path": image_path, "age": age, "gender": gender})
        return True

    def search_similar(self, query_image, top_k=3):
        """Tìm kiếm ảnh tương tự"""
        processed = self.preprocess_image(query_image)
        features = self.extract_features(processed)
        query_vec = np.concatenate([
            features["lbp"],
            features["geometry"],
            features["color"]
        ]).astype(np.float32)
        
        # Tính khoảng cách
        distances = []
        for vec in self.features:
            dist = np.linalg.norm(vec - query_vec)
            distances.append(dist)
        
        # Lấy top K kết quả
        sorted_indices = np.argsort(distances)[:top_k]
        return [(self.metadata[i], distances[i]) for i in sorted_indices]
      
    def load_dataset(self, dataset_path="dataset"):
      """Tự động load ảnh theo định dạng filename của bạn"""
      gender_mapping = {'0': 'male', '1': 'female'}  # Ánh xạ mã giới tính
      
      for root, _, files in os.walk(dataset_path):
          for filename in files:
              if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                  filepath = os.path.join(root, filename)
                  
                  try:
                      # Phân tích filename
                      base_name = os.path.splitext(filename)[0]
                      parts = base_name.split('_')
                      
                      # Xử lý metadata
                      age = None
                      gender = None
                      
                      if len(parts) >= 4:  # Đảm bảo đủ các phần
                          # Xử lý tuổi
                          try:
                              age = int(parts[0])
                          except ValueError:
                              pass
                          
                          # Xử lý giới tính
                          gender_code = parts[1]
                          gender = gender_mapping.get(gender_code, None)
                      
                      # Thêm vào database nếu chưa tồn tại
                      if not any(meta['path'] == filepath for meta in self.metadata):
                          self.add_to_database(filepath, age, gender)
                          print(f"✅ Đã thêm: {filename} | Tuổi: {age or 'N/A'} | Giới tính: {gender or 'N/A'}")
                      else:
                          print(f"⏩ Đã tồn tại: {filename}")
                          
                  except Exception as e:
                      print(f"❌ Lỗi xử lý {filename}: {str(e)}")
                    

class Application(tk.Tk):
    def __init__(self, system):
        super().__init__()
        self.system = system
        self.title("Hệ thống nhận dạng khuôn mặt người cao tuổi")
        self.geometry("800x600")
        
        # Giao diện
        self.create_widgets()
        
    def create_widgets(self):
        """Xây dựng giao diện người dùng"""
        # Frame chính
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Nút chọn ảnh
        self.btn_select = ttk.Button(main_frame, text="Chọn ảnh", command=self.select_image)
        self.btn_select.pack(pady=5)
        
        # Hiển thị ảnh đầu vào
        self.input_image_label = ttk.Label(main_frame)
        self.input_image_label.pack()
        
        # Nút tìm kiếm
        self.btn_search = ttk.Button(main_frame, text="Tìm kiếm", 
                                   command=self.perform_search, state=tk.DISABLED)
        self.btn_search.pack(pady=5)
        
        # Kết quả
        self.result_frame = ttk.Frame(main_frame)
        self.result_frame.pack(fill=tk.BOTH, expand=True)
        
        # Thanh tiến trình
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        
    def select_image(self):
        """Xử lý chọn ảnh"""
        file_path = filedialog.askopenfilename(filetypes=[
            ("Image files", "*.jpg *.jpeg *.png")])
        
        if file_path:
            self.current_image = cv2.imread(file_path)
            self.show_image(self.current_image, self.input_image_label)
            self.btn_search["state"] = tk.NORMAL
            
    def perform_search(self):
        """Thực hiện tìm kiếm và hiển thị kết quả"""
        self.progress.start()
        results = self.system.search_similar(self.current_image)
        self.progress.stop()
        
        # Xóa kết quả cũ
        for widget in self.result_frame.winfo_children():
            widget.destroy()
            
        # Hiển thị kết quả
        for idx, (meta, dist) in enumerate(results):
            frame = ttk.Frame(self.result_frame)
            frame.pack(side=tk.LEFT, padx=5)
            
            # Hiển thị ảnh
            img = cv2.imread(meta["path"])
            self.show_image(img, frame)
            
            # Hiển thị thông tin
            info = f"Độ tương đồng: {dist:.2f}\nTuổi: {meta.get('age','N/A')}\nGiới tính: {meta.get('gender','N/A')}"
            ttk.Label(frame, text=info).pack()
    
    def show_image(self, image, parent):
        """Hiển thị ảnh trên GUI"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image.thumbnail((200, 200))
        photo = ImageTk.PhotoImage(image)
        
        label = ttk.Label(parent)
        label.image = photo
        label.configure(image=photo)
        label.pack()

    
                    
if __name__ == "__main__":
    # Khởi tạo hệ thống
    face_system = ElderlyFaceSystem()
    
    # Tự động load dataset
    face_system.load_dataset()
    
    # Chạy ứng dụng
    app = Application(face_system)
    app.mainloop()