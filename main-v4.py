import os
import time
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import ImageTk

class ElderlyFaceSearchSystem:
    def __init__(self, data_dir='dataset', features_file='face_features.json'):
        """
        Khởi tạo hệ thống nhận dạng khuôn mặt người cao tuổi
        
        Args:
            data_dir: Thư mục chứa dữ liệu ảnh
        """
        self.data_dir = data_dir
        self.features_file = features_file
        self.image_size = (224, 224)  # Kích thước chuẩn cho ảnh
        self.face_features_db = []  # Cơ sở dữ liệu lưu trữ đặc trưng
        self.images_paths = []  # Đường dẫn đến ảnh
        self.feature_extraction_times = {}  # Thời gian trích xuất đặc trưng của từng ảnh
        
        # Tạo thư mục nếu chưa tồn tại
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # Kiểm tra xem có file đặc trưng đã lưu không
        if os.path.exists(features_file):
            self.load_features_from_file()
        else:
            # Nếu không có file đặc trưng, trích xuất đặc trưng từ ảnh
            self.load_database()
       
    def load_features_from_file(self):
        """Tải đặc trưng từ file JSON"""
        print(f"Đang tải đặc trưng từ file {self.features_file}...")
        
        try:
            with open(self.features_file, 'r') as f:
                data = json.load(f)
                
            # Kiểm tra phiên bản dữ liệu nếu cần
            if 'version' in data and data['version'] == '1.0':
                # Chuyển đổi lại dữ liệu từ danh sách sang numpy array
                self.face_features_db = [np.array(features) for features in data['features']]
                self.images_paths = data['image_paths']
                self.feature_extraction_times = data.get('extraction_times', {})
                
                print(f"Đã tải thành công {len(self.face_features_db)} vector đặc trưng.")
            else:
                print("Phiên bản dữ liệu không khớp, cần trích xuất lại đặc trưng.")
                self.load_database()
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Lỗi khi tải file đặc trưng: {e}")
            self.load_database()
    
    def save_features_to_file(self):
        """Lưu đặc trưng vào file JSON"""
        print(f"Lưu đặc trưng vào file {self.features_file}...")
        
        # Chuyển đổi numpy array sang danh sách để có thể serialize
        features_list = [features.tolist() for features in self.face_features_db]
        
        data = {
            'version': '1.0',
            'features': features_list,
            'image_paths': self.images_paths,
            'extraction_times': self.feature_extraction_times
        }
        
        with open(self.features_file, 'w') as f:
            json.dump(data, f)
            
        print(f"Đã lưu {len(self.face_features_db)} vector đặc trưng vào file.")
    
    def load_database(self):
        """Tải dữ liệu ảnh và trích xuất đặc trưng"""
        print("Đang tải dữ liệu và trích xuất đặc trưng...")
        
        # Reset danh sách nếu cần
        self.face_features_db = []
        self.images_paths = []
        self.feature_extraction_times = {}
        
        # Danh sách các ảnh trong thư mục
        all_images = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    all_images.append(image_path)
        
        # Số lượng ảnh đã xử lý
        total_images = len(all_images)
        processed = 0
        
        # Duyệt qua tất cả các ảnh trong thư mục
        for image_path in all_images:
            try:
                # Đo thời gian trích xuất đặc trưng
                start_time = time.time()
                
                # Đọc ảnh
                img = cv2.imread(image_path)
                if img is None:
                    continue
                    
                # Tiền xử lý ảnh
                preprocessed_img = self.preprocess_image(img)
                
                # Trích xuất đặc trưng
                features = self.extract_features(preprocessed_img)
                
                # Tính thời gian trích xuất
                extraction_time = time.time() - start_time
                self.feature_extraction_times[image_path] = extraction_time
                
                # Lưu trữ đặc trưng và đường dẫn ảnh
                self.face_features_db.append(features)
                self.images_paths.append(image_path)
                
                # Cập nhật tiến độ
                processed += 1
                if processed % 10 == 0 or processed == total_images:
                    print(f"Đã xử lý {processed}/{total_images} ảnh ({processed/total_images*100:.1f}%)")
                    
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {image_path}: {e}")
        
        print(f"Đã tải và trích xuất đặc trưng của {len(self.face_features_db)} ảnh")
        
        # Lưu đặc trưng vào file
        self.save_features_to_file()
    
    def update_database(self):
        """Cập nhật CSDL với các ảnh mới được thêm vào thư mục"""
        print("Kiểm tra và cập nhật CSDL với các ảnh mới...")
        
        # Lấy danh sách tất cả các ảnh trong thư mục
        current_images = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    current_images.append(image_path)
        
        # Tìm các ảnh mới (chưa có trong CSDL)
        existing_images = set(self.images_paths)
        new_images = [img for img in current_images if img not in existing_images]
        
        if not new_images:
            print("Không có ảnh mới cần cập nhật")
            return
        
        print(f"Phát hiện {len(new_images)} ảnh mới cần cập nhật")
        
        # Xử lý từng ảnh mới
        for image_path in new_images:
            try:
                # Đo thời gian trích xuất đặc trưng
                start_time = time.time()
                
                # Đọc ảnh
                img = cv2.imread(image_path)
                if img is None:
                    continue
                    
                # Tiền xử lý ảnh
                preprocessed_img = self.preprocess_image(img)
                
                # Trích xuất đặc trưng
                features = self.extract_features(preprocessed_img)
                
                # Tính thời gian trích xuất
                extraction_time = time.time() - start_time
                self.feature_extraction_times[image_path] = extraction_time
                
                # Lưu trữ đặc trưng và đường dẫn ảnh
                self.face_features_db.append(features)
                self.images_paths.append(image_path)
                
                print(f"Đã xử lý ảnh mới: {image_path}")
                    
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {image_path}: {e}")
        
        # Lưu đặc trưng vào file
        self.save_features_to_file()
        print(f"Đã cập nhật CSDL với {len(new_images)} ảnh mới")
    
    def check_for_modified_images(self):
        """Kiểm tra xem có ảnh nào bị thay đổi không dựa vào thời gian sửa đổi"""
        print("Kiểm tra các ảnh đã thay đổi...")
        
        modified_images = []
        missing_images = []
        
        # Kiểm tra từng ảnh trong CSDL
        for image_path in self.images_paths:
            if not os.path.exists(image_path):
                missing_images.append(image_path)
                continue
                
            # Lấy thời gian sửa đổi của file
            mod_time = os.path.getmtime(image_path)
            
            # So sánh với thời gian trích xuất đặc trưng
            if image_path in self.feature_extraction_times:
                if mod_time > self.feature_extraction_times[image_path]:
                    modified_images.append(image_path)
            else:
                # Nếu không có thông tin về thời gian trích xuất, thêm vào danh sách cần cập nhật
                modified_images.append(image_path)
        
        return modified_images, missing_images
    
    def update_modified_images(self):
        """Cập nhật các ảnh đã bị thay đổi"""
        modified_images, missing_images = self.check_for_modified_images()
        
        # Xóa các ảnh không còn tồn tại khỏi CSDL
        if missing_images:
            print(f"Xóa {len(missing_images)} ảnh không còn tồn tại khỏi CSDL")
            for img_path in missing_images:
                idx = self.images_paths.index(img_path)
                self.images_paths.pop(idx)
                self.face_features_db.pop(idx)
                if img_path in self.feature_extraction_times:
                    del self.feature_extraction_times[img_path]
        
        # Cập nhật các ảnh đã thay đổi
        if modified_images:
            print(f"Cập nhật {len(modified_images)} ảnh đã thay đổi")
            for img_path in modified_images:
                try:
                    # Đọc ảnh
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                        
                    # Tiền xử lý ảnh
                    preprocessed_img = self.preprocess_image(img)
                    
                    # Trích xuất đặc trưng
                    start_time = time.time()
                    features = self.extract_features(preprocessed_img)
                    extraction_time = time.time() - start_time
                    
                    # Cập nhật đặc trưng trong CSDL
                    idx = self.images_paths.index(img_path)
                    self.face_features_db[idx] = features
                    self.feature_extraction_times[img_path] = extraction_time
                    
                    print(f"Đã cập nhật đặc trưng cho ảnh: {img_path}")
                        
                except Exception as e:
                    print(f"Lỗi khi cập nhật ảnh {img_path}: {e}")
        
        # Nếu có thay đổi, lưu lại CSDL
        if modified_images or missing_images:
            self.save_features_to_file()
            print("Đã cập nhật và lưu CSDL")
        else:
            print("Không có ảnh nào cần cập nhật")
    
    def export_to_csv(self, csv_file='face_features.csv'):
        """Xuất đặc trưng ra file CSV"""
        import csv
        
        print(f"Xuất đặc trưng ra file CSV: {csv_file}")
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Ghi header
            header = ['image_path']
            # Thêm số lượng cột đặc trưng
            if self.face_features_db:
                feature_count = len(self.face_features_db[0])
                for i in range(feature_count):
                    header.append(f'feature_{i}')
            
            writer.writerow(header)
            
            # Ghi dữ liệu
            for i, features in enumerate(self.face_features_db):
                row = [self.images_paths[i]] + features.tolist()
                writer.writerow(row)
        
        print(f"Đã xuất {len(self.face_features_db)} dòng dữ liệu ra file CSV")
    
    def import_from_csv(self, csv_file='face_features.csv'):
        """Nhập đặc trưng từ file CSV"""
        import csv
        
        print(f"Nhập đặc trưng từ file CSV: {csv_file}")
        
        try:
            with open(csv_file, 'r', newline='') as f:
                reader = csv.reader(f)
                
                # Đọc header
                header = next(reader)
                
                # Reset dữ liệu hiện tại
                self.face_features_db = []
                self.images_paths = []
                
                # Đọc từng dòng dữ liệu
                for row in reader:
                    if not row:
                        continue
                        
                    image_path = row[0]
                    # Chuyển đổi các cột còn lại thành vector đặc trưng
                    features = np.array([float(x) for x in row[1:]])
                    
                    self.images_paths.append(image_path)
                    self.face_features_db.append(features)
            
            print(f"Đã nhập {len(self.face_features_db)} vector đặc trưng từ file CSV")
            
            # Lưu lại vào file JSON để cập nhật định dạng
            self.save_features_to_file()
            
        except Exception as e:
            print(f"Lỗi khi nhập từ file CSV: {e}")
    
    # --- Các phương thức còn lại từ lớp gốc ---
    
    
        """
        Trích xuất đặc trưng dựa trên hình dạng
        
        Args:
            gray_image: Ảnh thang xám
            
        Returns:
            Vector đặc trưng hình dạng
        """
        # Phát hiện các điểm đặc trưng trên khuôn mặt
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        # nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')
        mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Phát hiện khuôn mặt
        faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
        
        # Nếu không tìm thấy khuôn mặt, trả về vector rỗng
        if len(faces) == 0:
            return np.zeros(10)
        
        # Lấy khuôn mặt lớn nhất
        face_x, face_y, face_w, face_h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        
        # Phát hiện mắt trong vùng phía trên khuôn mặt
        face_upper = gray_image[face_y:face_y+face_h//2, face_x:face_x+face_w]
        eyes = eye_cascade.detectMultiScale(face_upper, 1.3, 5)
        
        # Phát hiện mũi trong vùng giữa khuôn mặt
        # face_middle = gray_image[face_y+face_h//4:face_y+3*face_h//4, face_x:face_x+face_w]
        # noses = nose_cascade.detectMultiScale(face_middle, 1.3, 5)
        
        # Phát hiện miệng trong vùng dưới khuôn mặt
        face_lower = gray_image[face_y+face_h//2:face_y+face_h, face_x:face_x+face_w]
        mouths = mouth_cascade.detectMultiScale(face_lower, 1.8, 11)
        
        # Tính toán các tỷ lệ
        shape_features = []
        
        # Tỷ lệ chiều cao/chiều rộng khuôn mặt
        face_ratio = face_h / max(1, face_w)
        shape_features.append(face_ratio)
        
        # Độ tương phản trung bình (đo độ sâu của nếp nhăn)
        gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        avg_gradient = np.mean(gradient_magnitude)
        shape_features.append(avg_gradient)
        
        # Vị trí tương đối của mắt, mũi, miệng nếu phát hiện được
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])
            left_eye_x, left_eye_y, left_eye_w, left_eye_h = eyes[0]
            right_eye_x, right_eye_y, right_eye_w, right_eye_h = eyes[1]
            
            # Khoảng cách giữa hai mắt
            eye_distance = (right_eye_x - left_eye_x) / face_w
            shape_features.append(eye_distance)
            
            # Diện tích mắt (có thể nhỏ hơn ở người cao tuổi)
            eye_area_ratio = (left_eye_w * left_eye_h + right_eye_w * right_eye_h) / (face_w * face_h)
            shape_features.append(eye_area_ratio)
        else:
            shape_features.extend([0, 0])  # Không phát hiện được mắt
        
        # if len(noses) > 0:
        #     nose_x, nose_y, nose_w, nose_h = sorted(noses, key=lambda n: n[2] * n[3], reverse=True)[0]
            
        #     # Vị trí tương đối của mũi
        #     nose_position = (nose_y + nose_h/2) / face_h
        #     shape_features.append(nose_position)
        # else:
        #     shape_features.append(0.5)  # Giá trị mặc định
        
        if len(mouths) > 0:
            mouth_x, mouth_y, mouth_w, mouth_h = sorted(mouths, key=lambda m: m[2] * m[3], reverse=True)[0]
            
            # Vị trí tương đối của miệng
            mouth_position = (mouth_y + face_h//2) / face_h
            shape_features.append(mouth_position)
            
            # Chiều rộng miệng so với khuôn mặt
            mouth_width_ratio = mouth_w / face_w
            shape_features.append(mouth_width_ratio)
        else:
            shape_features.extend([0.7, 0.5])  # Giá trị mặc định
        
        # Phân tích nếp nhăn bằng cách tính gradient cục bộ trong các vùng quan tâm
        
        # Vùng trán
        forehead = gray_image[face_y:face_y+face_h//4, face_x:face_x+face_w]
        forehead_gradient = np.mean(cv2.Sobel(forehead, cv2.CV_64F, 0, 1, ksize=3))
        shape_features.append(forehead_gradient)
        
        # Vùng quanh mắt
        eye_region = gray_image[face_y+face_h//4:face_y+face_h//2, face_x:face_x+face_w]
        eye_region_gradient = np.mean(cv2.Sobel(eye_region, cv2.CV_64F, 1, 1, ksize=3))
        shape_features.append(eye_region_gradient)
        
        # Vùng quanh miệng
        mouth_region = gray_image[face_y+2*face_h//3:face_y+face_h, face_x:face_x+face_w]
        mouth_region_gradient = np.mean(cv2.Sobel(mouth_region, cv2.CV_64F, 1, 1, ksize=3))
        shape_features.append(mouth_region_gradient)
        
        return np.array(shape_features)
    
    def preprocess_image(self, image):
        raise NotImplementedError("Bạn cần cài đặt hàm tiền xử lý ảnh")

    def extract_features(self, image):
        raise NotImplementedError("Bạn cần cài đặt hàm trích xuất đặc trưng")
      
    def search_similar_faces(self, query_image, top_k=3):
        """
        Tìm kiếm ảnh mặt người cao tuổi tương tự
        
        Args:
            query_image: Ảnh đầu vào
            top_k: Số lượng kết quả trả về
            
        Returns:
            Danh sách đường dẫn ảnh và độ tương đồng
        """
        # Tiền xử lý ảnh query
        preprocessed_img = self.preprocess_image(query_image)
        
        # Trích xuất đặc trưng
        query_features = self.extract_features(preprocessed_img)
        
        # Tính độ tương đồng với tất cả ảnh trong CSDL
        similarities = []
        
        for i, features in enumerate(self.face_features_db):
            # Tính khoảng cách cosine
            similarity = self.cosine_similarity(query_features, features)
            # min_len = min(len(query_features), len(features))
            # similarity = self.cosine_similarity(query_features[:min_len], features[:min_len])
            similarities.append((similarity, self.images_paths[i]))
        
        # Sắp xếp theo độ tương đồng giảm dần
        similarities.sort(reverse=True)
        
        # Trả về top_k kết quả
        return similarities[:top_k]
    
    def cosine_similarity(self, a, b):
        """
        Tính độ tương đồng cosine giữa hai vector
        
        Args:
            a, b: Hai vector đặc trưng
            
        Returns:
            Độ tương đồng cosine
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0
        
        return dot_product / (norm_a * norm_b)
    
    def chi_square_distance(self, a, b):
        """
        Tính khoảng cách chi-square giữa hai vector
        
        Args:
            a, b: Hai vector đặc trưng
            
        Returns:
            Khoảng cách chi-square
        """
        # Tránh chia cho 0
        sum_ab = a + b
        valid_indices = sum_ab != 0
        
        # Tính khoảng cách chỉ trên các phần tử hợp lệ
        distance = np.sum(((a[valid_indices] - b[valid_indices]) ** 2) / sum_ab[valid_indices])
        
        return distance
    
    def euclidean_distance(self, a, b):
        """
        Tính khoảng cách Euclid giữa hai vector
        
        Args:
            a, b: Hai vector đặc trưng
            
        Returns:
            Khoảng cách Euclid
        """
        return np.sqrt(np.sum((a - b) ** 2))
    
    def display_results(self, query_image, similar_images):
        """
        Hiển thị kết quả tìm kiếm
        
        Args:
            query_image: Ảnh truy vấn
            similar_images: Danh sách (độ tương đồng, đường dẫn ảnh)
        """
        # Tạo figure với 2 hàng, 2 cột
        plt.figure(figsize=(12, 10))
        
        # Hiển thị ảnh truy vấn
        plt.subplot(2, 2, 1)
        query_img_rgb = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
        plt.imshow(query_img_rgb)
        plt.title('Ảnh truy vấn')
        plt.axis('off')
        
        # Hiển thị các ảnh tương tự
        for i, (similarity, img_path) in enumerate(similar_images):
            plt.subplot(2, 2, i+2)
            similar_img = cv2.imread(img_path)
            similar_img_rgb = cv2.cvtColor(similar_img, cv2.COLOR_BGR2RGB)
            plt.imshow(similar_img_rgb)
            plt.title(f'Tương tự #{i+1}, Độ tương đồng: {similarity:.4f}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def create_gui(self):
        """Tạo giao diện người dùng đơn giản với Tkinter"""
        self.root = tk.Tk()
        self.root.title("Hệ thống tìm kiếm khuôn mặt người cao tuổi")
        self.root.geometry("800x600")
        
        # Frame để chứa các thành phần
        frame = tk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Label hướng dẫn
        tk.Label(frame, text="Hệ thống tìm kiếm ảnh mặt người cao tuổi", font=("Arial", 16)).pack(pady=10)
        tk.Label(frame, text="Chọn ảnh để tìm kiếm các khuôn mặt tương tự", font=("Arial", 12)).pack(pady=5)
        
        # Button để chọn ảnh
        self.select_button = Button(frame, text="Chọn ảnh", command=self.select_image)
        self.select_button.pack(pady=10)
        
        # Frame để hiển thị ảnh đầu vào
        self.input_frame = tk.Frame(frame)
        self.input_frame.pack(pady=10)
        
        tk.Label(self.input_frame, text="Ảnh đầu vào:").pack()
        self.input_image_label = Label(self.input_frame)
        self.input_image_label.pack()
        
        # Button để tìm kiếm
        self.search_button = Button(frame, text="Tìm kiếm ảnh tương tự", command=self.search_button_click)
        self.search_button.pack(pady=10)
        self.search_button["state"] = "disabled"
        
        # Frame để hiển thị kết quả
        self.results_frame = tk.Frame(frame)
        self.results_frame.pack(pady=10, fill=tk.X)
        
        tk.Label(self.results_frame, text="Kết quả tìm kiếm:").pack()
        
        # Frame để chứa 3 ảnh kết quả
        self.result_images_frame = tk.Frame(self.results_frame)
        self.result_images_frame.pack()
        
        # Labels để hiển thị ảnh kết quả
        self.result_labels = []
        self.result_similarity_labels = []
        
        for i in range(3):
            result_frame = tk.Frame(self.result_images_frame)
            result_frame.pack(side=tk.LEFT, padx=10)
            
            result_label = Label(result_frame)
            result_label.pack()
            self.result_labels.append(result_label)
            
            similarity_label = Label(result_frame, text="")
            similarity_label.pack()
            self.result_similarity_labels.append(similarity_label)
        
        # Biến để lưu trữ ảnh đầu vào hiện tại
        self.current_input_image = None
        
        self.root.mainloop()
    
    def select_image(self):
        """Hàm xử lý sự kiện khi người dùng chọn ảnh"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            # Đọc ảnh
            self.current_input_image = cv2.imread(file_path)
            
            if self.current_input_image is not None:
                # Hiển thị ảnh đầu vào
                self.display_input_image()
                
                # Kích hoạt nút tìm kiếm
                self.search_button["state"] = "normal"
    
    def display_input_image(self):
        """Hiển thị ảnh đầu vào trên giao diện"""
        if self.current_input_image is None:
            return
        
        # Resize ảnh để hiển thị
        h, w = self.current_input_image.shape[:2]
        max_size = 200
        scale = min(max_size/h, max_size/w)
        display_img = cv2.resize(self.current_input_image, (int(w*scale), int(h*scale)))
        
        # Chuyển đổi sang định dạng để hiển thị trên Tkinter
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display_img)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Lưu trữ tham chiếu để tránh bị thu hồi bởi garbage collector
        self.input_image_label.image = img_tk
        self.input_image_label.configure(image=img_tk)
    
    def search_button_click(self):
        """Hàm xử lý sự kiện khi người dùng nhấn nút tìm kiếm"""
        if self.current_input_image is None:
            return
        
        # Tìm kiếm ảnh tương tự
        similar_images = self.search_similar_faces(self.current_input_image)
        
        # Hiển thị kết quả
        self.display_result_images(similar_images)
    
    def display_result_images(self, similar_images):
        """Hiển thị các ảnh kết quả tìm kiếm trên giao diện"""
        # Xóa các ảnh kết quả cũ
        for label in self.result_labels:
            label.configure(image=None)
        
        for label, similarity_label, (similarity, img_path) in zip(self.result_labels, self.result_similarity_labels, similar_images):
            # Đọc ảnh kết quả
            result_img = cv2.imread(img_path)
            
            if result_img is not None:
                # Resize ảnh để hiển thị
                h, w = result_img.shape[:2]
                max_size = 150
                scale = min(max_size/h, max_size/w)
                display_img = cv2.resize(result_img, (int(w*scale), int(h*scale)))
                
                # Chuyển đổi sang định dạng để hiển thị trên Tkinter
                display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(display_img)
                img_tk = ImageTk.PhotoImage(image=img)
                
                # Lưu trữ tham chiếu để tránh bị thu hồi bởi garbage collector
                label.image = img_tk
                label.configure(image=img_tk)
                
                # Hiển thị độ tương đồng
                similarity_label.configure(text=f"Độ tương đồng: {similarity:.4f}")


# Đánh giá hệ thống
def evaluate_system(system, test_images_dir):
    """
    Đánh giá hiệu suất của hệ thống nhận dạng khuôn mặt người cao tuổi
    
    Args:
        system: Hệ thống nhận dạng
        test_images_dir: Thư mục chứa ảnh kiểm thử
    """
    # Danh sách ảnh kiểm thử
    test_images = []
    for root, _, files in os.walk(test_images_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(os.path.join(root, file))
    
    if not test_images:
        print("Không tìm thấy ảnh kiểm thử")
        return
    
    print(f"Đánh giá trên {len(test_images)} ảnh kiểm thử")
    
    # Các chỉ số đánh giá
    avg_time = 0
    
    for img_path in test_images:
        # Đọc ảnh
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Đo thời gian tìm kiếm
        start_time = time.time()
        similar_images = system.search_similar_faces(img)
        end_time = time.time()
        
        # Cập nhật thời gian trung bình
        avg_time += (end_time - start_time)
    
    # Tính thời gian trung bình
    avg_time /= len(test_images)
    
    print(f"Thời gian tìm kiếm trung bình: {avg_time:.4f} giây")


# Demo chính
def main():
    # Khởi tạo hệ thống
    system = ElderlyFaceSearchSystem(data_dir='elderly_faces_dataset')
    
    # Chọn chế độ demo
    print("Chọn chế độ demo:")
    print("1. Demo với giao diện đồ họa")
    print("2. Demo với đánh giá hiệu suất")
    
    choice = input("Nhập lựa chọn của bạn (1/2): ")
    
    if choice == '1':
        # Demo với giao diện đồ họa
        system.create_gui()
    elif choice == '2':
        # Demo với đánh giá hiệu suất
        test_dir = input("Nhập đường dẫn đến thư mục ảnh kiểm thử: ")
        evaluate_system(system, test_dir)
    else:
        print("Lựa chọn không hợp lệ")

if __name__ == "__main__":
    main()