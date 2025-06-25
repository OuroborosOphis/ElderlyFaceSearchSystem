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
from skimage.feature import hog, local_binary_pattern

class ElderlyFaceSearchSystem:
    def __init__(self, data_dir='dataset', features_file='face_features.json'):
        """
        Khởi tạo hệ thống tìm kiếm ảnh mặt người cao tuổi
        - data_dir: thư mục chứa dữ liệu ảnh
        - features_file: file JSON lưu trữ đặc trưng đã trích xuất
        """
        self.data_dir = data_dir
        self.features_file = features_file
        self.image_size = (224, 224)  # Kích thước chuẩn hóa ảnh
        self.face_features_db = []  # Lưu trữ vector đặc trưng
        self.images_paths = []  # Lưu đường dẫn ảnh tương ứng
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # Tải đặc trưng từ file nếu có, ngược lại trích xuất lại
        if os.path.exists(features_file):
            self.load_features_from_file()
        else:
            self.load_database()
    
    def load_features_from_file(self):
        """
        Tải các vector đặc trưng đã được lưu trữ từ file JSON
        Giúp tiết kiệm thời gian không phải trích xuất lại
        """
        print(f"Đang tải đặc trưng từ file {self.features_file}...")
        
        try:
            with open(self.features_file, 'r') as f:
                data = json.load(f)
                
            # Kiểm tra phiên bản để đảm bảo tương thích
            if 'version' in data and data['version'] == '2.0':
                self.face_features_db = [np.array(features) for features in data['features']]
                self.images_paths = data['image_paths']
                
                print(f"Đã tải thành công {len(self.face_features_db)} vector đặc trưng.")
            else:
                print("Phiên bản dữ liệu không khớp, cần trích xuất lại đặc trưng.")
                self.load_database()
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Lỗi khi tải file đặc trưng: {e}")
            self.load_database()
    
    def save_features_to_file(self):
        """
        Lưu các vector đặc trưng vào file JSON để sử dụng lại
        """
        print(f"Lưu đặc trưng vào file {self.features_file}...")
        
        # Chuyển numpy arrays thành lists để có thể serialize JSON
        features_list = [features.tolist() for features in self.face_features_db]
        
        data = {
            'version': '2.0',  # Phiên bản mới phù hợp với báo cáo
            'features': features_list,
            'image_paths': self.images_paths
        }
        
        with open(self.features_file, 'w') as f:
            json.dump(data, f)
            
        print(f"Đã lưu {len(self.face_features_db)} vector đặc trưng vào file.")
    
    def load_database(self):
        """
        Giai đoạn 1: Thu thập, xử lý, trích xuất đặc trưng và lưu trữ ảnh
        Thực hiện các bước 1-4 như mô tả trong báo cáo
        """
        print("Đang tải dữ liệu và trích xuất đặc trưng...")
        
        self.face_features_db = []
        self.images_paths = []
        
        # Bước 1: Thu thập ảnh chân dung người cao tuổi
        all_images = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    all_images.append(image_path)
        
        total_images = len(all_images)
        processed = 0
        
        print(f"Tìm thấy {total_images} ảnh trong cơ sở dữ liệu")
        
        for image_path in all_images:
            try:
                # Bước 2: Tiền xử lý ảnh
                img = cv2.imread(image_path)
                if img is None:
                    continue
                    
                preprocessed_img = self.preprocess_image(img)
                
                # Bước 3: Trích xuất đặc trưng (LBP + HOG)
                features = self.extract_features(preprocessed_img)
                
                # Bước 4: Lưu trữ đặc trưng
                self.face_features_db.append(features)
                self.images_paths.append(image_path)
                
                processed += 1
                if processed % 10 == 0 or processed == total_images:
                    print(f"Đã xử lý {processed}/{total_images} ảnh ({processed/total_images*100:.1f}%)")
                    
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {image_path}: {e}")
        
        print(f"Hoàn thành trích xuất đặc trưng cho {len(self.face_features_db)} ảnh")
        
        # Lưu vào file JSON
        self.save_features_to_file()
    
    def preprocess_image(self, image):
        """
        Bước 2: Tiền xử lý ảnh
        - Phát hiện khuôn mặt bằng Cascade Classifier
        - Cắt vùng mặt và resize về kích thước chuẩn
        - Đảm bảo các ảnh có cùng kích thước và tỉ lệ khung hình
        """
        # Chuyển sang ảnh xám để phát hiện khuôn mặt
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sử dụng Cascade Classifier để detect khuôn mặt như trong báo cáo
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Nếu không tìm thấy khuôn mặt, resize toàn bộ ảnh
        if len(faces) == 0:
            resized = cv2.resize(image, self.image_size)
            return resized
        
        # Chọn khuôn mặt lớn nhất (khuôn mặt đầu tiên phát hiện được)
        (x, y, w, h) = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        
        # Crop vùng mặt với một chút padding
        face_region = image[max(0, y-10):min(image.shape[0], y+h+10), 
                          max(0, x-10):min(image.shape[1], x+w+10)]
        
        # Resize về kích thước chuẩn
        resized = cv2.resize(face_region, self.image_size)
        
        return resized
    
    def extract_features(self, image):
        """
        Bước 3: Trích xuất đặc trưng
        Kết hợp LBP và HOG như mô tả trong báo cáo
        """
        # Chuyển sang ảnh xám cho việc trích xuất đặc trưng
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Trích xuất đặc trưng LBP
        lbp_features = self.extract_lbp_features(gray)
        
        # Trích xuất đặc trưng HOG
        hog_features = self.extract_hog_features(gray)
        
        # Kết hợp hai loại đặc trưng
        combined_features = np.concatenate((lbp_features, hog_features))
        
        # Chuẩn hóa vector đặc trưng
        norm = np.linalg.norm(combined_features)
        if norm > 0:
            combined_features = combined_features / norm
        
        return combined_features
    
    def extract_lbp_features(self, gray_image):
        """
        Trích xuất đặc trưng LBP (Local Binary Patterns)
        Phù hợp với đặc điểm tuổi già - nếp nhăn và kết cấu da
        """
        # Chia ảnh thành các cell 8x8 như trong HOG
        h, w = gray_image.shape
        cell_h, cell_w = h // 8, w // 8
        
        lbp_histograms = []
        
        # Tính LBP cho từng cell
        for i in range(8):
            for j in range(8):
                # Lấy cell hiện tại
                start_y, end_y = i * cell_h, min((i + 1) * cell_h, h)
                start_x, end_x = j * cell_w, min((j + 1) * cell_w, w)
                cell = gray_image[start_y:end_y, start_x:end_x]
                
                if cell.size == 0:
                    continue
                
                # Tính LBP cho cell này
                # Sử dụng 8 pixel láng giềng, bán kính = 1
                radius = 1
                n_points = 8
                lbp = local_binary_pattern(cell, n_points, radius, method='uniform')
                
                # Tính histogram của LBP
                hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, 
                                    range=(0, n_points + 2))
                
                # Chuẩn hóa histogram
                if np.sum(hist) > 0:
                    hist = hist.astype(float) / np.sum(hist)
                
                lbp_histograms.append(hist)
        
        # Ghép nối tất cả histogram thành vector đặc trưng LBP
        lbp_features = np.concatenate(lbp_histograms)
        
        return lbp_features
    
    def extract_hog_features(self, gray_image):
        """
        Trích xuất đặc trưng HOG (Histogram of Oriented Gradients)
        Mô tả chi tiết cấu trúc hình dạng khuôn mặt
        """
        # Tham số HOG theo mô tả trong báo cáo
        # - Cell: 8x8 pixels
        # - 9 hướng gradient 
        # - Block chuẩn hóa để tăng độ ổn định
        
        hog_features = hog(
            gray_image,
            orientations=9,           # 9 hướng gradient
            pixels_per_cell=(8, 8),   # Cell 8x8 pixels như mô tả
            cells_per_block=(2, 2),   # Block 2x2 cells để chuẩn hóa
            block_norm='L2-Hys',      # Chuẩn hóa L2-Hys
            feature_vector=True       # Trả về vector đặc trưng
        )
        
        return hog_features
    
    def search_similar_faces(self, query_image, top_k=3):
        """
        Giai đoạn 2: Tìm kiếm 3 ảnh tương đồng (bước 5-10)
        """
        # Bước 6: Tiền xử lý ảnh đầu vào
        preprocessed_img = self.preprocess_image(query_image)
        
        # Bước 7: Trích xuất đặc trưng cho ảnh đầu vào
        query_features = self.extract_features(preprocessed_img)
        
        # Bước 8: Tính khoảng cách Euclidean với từng ảnh trong hệ thống
        distances = []
        
        for i, features in enumerate(self.face_features_db):
            # Sử dụng khoảng cách Euclidean như mô tả trong báo cáo
            distance = self.euclidean_distance(query_features, features)
            distances.append((distance, self.images_paths[i]))
        
        # Bước 9: Xếp hạng khoảng cách và trả về 3 khoảng cách nhỏ nhất
        # Khoảng cách càng nhỏ = càng giống
        distances.sort(key=lambda x: x[0])
        
        return distances[:top_k]
    
    def euclidean_distance(self, a, b):
        """
        Tính khoảng cách Euclidean như công thức trong báo cáo
        √(Σ(ai - bi)²)
        """
        return np.sqrt(np.sum((a - b) ** 2))
    
    def create_gui(self):
        """
        Tạo giao diện đồ họa đơn giản bằng Tkinter
        Bước 10: Hiển thị 3 ảnh giống nhất trên giao diện
        """
        self.root = tk.Tk()
        self.root.title("Hệ thống tìm kiếm khuôn mặt người cao tuổi")
        self.root.geometry("1200x800")
        
        # Frame chính
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Tiêu đề
        title_label = tk.Label(main_frame, 
                              text="Hệ thống tìm kiếm ảnh mặt người cao tuổi", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        subtitle_label = tk.Label(main_frame, 
                                 text="Chọn ảnh để tìm kiếm các khuôn mặt tương tự", 
                                 font=("Arial", 12))
        subtitle_label.pack(pady=5)
        
        # Nút chọn ảnh
        self.select_button = Button(main_frame, 
                                   text="Chọn ảnh đầu vào", 
                                   command=self.select_image,
                                   font=("Arial", 12))
        self.select_button.pack(pady=10)
        
        # Khung hiển thị ảnh đầu vào
        input_frame = tk.Frame(main_frame)
        input_frame.pack(pady=10)
        
        tk.Label(input_frame, text="Ảnh đầu vào:", font=("Arial", 12, "bold")).pack()
        self.input_image_label = Label(input_frame)
        self.input_image_label.pack(pady=5)
        
        # Nút tìm kiếm
        self.search_button = Button(main_frame, 
                                   text="Tìm kiếm ảnh tương tự", 
                                   command=self.search_button_click,
                                   font=("Arial", 12),
                                   state="disabled")
        self.search_button.pack(pady=10)
        
        # Khung hiển thị kết quả
        results_frame = tk.Frame(main_frame)
        results_frame.pack(pady=20, fill=tk.X)
        
        tk.Label(results_frame, text="Kết quả tìm kiếm (3 ảnh giống nhất):", 
                font=("Arial", 12, "bold")).pack()
        
        # Frame chứa 3 ảnh kết quả
        self.result_images_frame = tk.Frame(results_frame)
        self.result_images_frame.pack(pady=10)
        
        # Tạo 3 label để hiển thị kết quả
        self.result_labels = []
        self.result_distance_labels = []
        
        for i in range(3):
            result_frame = tk.Frame(self.result_images_frame)
            result_frame.pack(side=tk.LEFT, padx=15)
            
            # Label hiển thị thứ tự
            tk.Label(result_frame, text=f"#{i+1}", 
                    font=("Arial", 12, "bold")).pack()
            
            # Label hiển thị ảnh
            result_label = Label(result_frame)
            result_label.pack(pady=5)
            self.result_labels.append(result_label)
            
            # Label hiển thị khoảng cách
            distance_label = Label(result_frame, text="", font=("Arial", 10))
            distance_label.pack()
            self.result_distance_labels.append(distance_label)
        
        # Biến lưu ảnh đầu vào hiện tại
        self.current_input_image = None
        
        # Label hiển thị thống kê
        self.stats_label = tk.Label(main_frame, 
                                   text=f"Cơ sở dữ liệu: {len(self.face_features_db)} ảnh", 
                                   font=("Arial", 10))
        self.stats_label.pack(pady=(20, 0))
        
        self.root.mainloop()
    
    def select_image(self):
        """
        Bước 5: Nhận ảnh đầu vào
        """
        file_path = filedialog.askopenfilename(
            title="Chọn ảnh người cao tuổi",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            self.current_input_image = cv2.imread(file_path)
            
            if self.current_input_image is not None:
                self.display_input_image()
                self.search_button["state"] = "normal"
                
                # Xóa kết quả cũ
                self.clear_results()
    
    def display_input_image(self):
        """
        Hiển thị ảnh đầu vào trên giao diện
        """
        if self.current_input_image is None:
            return
        
        # Resize ảnh để hiển thị
        h, w = self.current_input_image.shape[:2]
        max_size = 200
        scale = min(max_size/h, max_size/w)
        display_img = cv2.resize(self.current_input_image, 
                               (int(w*scale), int(h*scale)))
        
        # Chuyển từ BGR sang RGB
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display_img)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Hiển thị ảnh
        self.input_image_label.image = img_tk
        self.input_image_label.configure(image=img_tk)
    
    def search_button_click(self):
        """
        Xử lý sự kiện nhấn nút tìm kiếm
        """
        if self.current_input_image is None:
            return
        
        print("Đang tìm kiếm ảnh tương tự...")
        start_time = time.time()
        
        # Thực hiện tìm kiếm
        similar_images = self.search_similar_faces(self.current_input_image)
        
        end_time = time.time()
        search_time = end_time - start_time
        
        print(f"Hoàn thành tìm kiếm trong {search_time:.3f} giây")
        
        # Hiển thị kết quả
        self.display_result_images(similar_images)
    
    def display_result_images(self, similar_images):
        """
        Bước 10: Hiển thị 3 ảnh giống nhất trên giao diện
        """
        # Xóa kết quả cũ
        self.clear_results()
        
        # Hiển thị 3 ảnh có khoảng cách nhỏ nhất
        for i, (distance, img_path) in enumerate(similar_images):
            if i >= 3:  # Chỉ hiển thị 3 ảnh đầu
                break
                
            result_img = cv2.imread(img_path)
            
            if result_img is not None:
                # Resize ảnh kết quả
                h, w = result_img.shape[:2]
                max_size = 150
                scale = min(max_size/h, max_size/w)
                display_img = cv2.resize(result_img, 
                                       (int(w*scale), int(h*scale)))
                
                # Chuyển BGR sang RGB
                display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(display_img)
                img_tk = ImageTk.PhotoImage(image=img)
                
                # Hiển thị ảnh
                self.result_labels[i].image = img_tk
                self.result_labels[i].configure(image=img_tk)
                
                # Hiển thị khoảng cách Euclidean
                self.result_distance_labels[i].configure(
                    text=f"Khoảng cách: {distance:.4f}")
    
    def clear_results(self):
        """
        Xóa kết quả hiển thị cũ
        """
        for label in self.result_labels:
            label.configure(image=None)
            label.image = None
        
        for label in self.result_distance_labels:
            label.configure(text="")


def evaluate_system_performance(system, test_images_dir):
    """
    Đánh giá hiệu suất hệ thống
    Tính thời gian tìm kiếm trung bình như mô tả trong báo cáo
    """
    test_images = []
    for root, _, files in os.walk(test_images_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(os.path.join(root, file))
    
    if not test_images:
        print("Không tìm thấy ảnh kiểm thử")
        return
    
    print(f"Đánh giá hiệu suất trên {len(test_images)} ảnh kiểm thử")
    
    total_time = 0
    successful_searches = 0
    
    for i, img_path in enumerate(test_images):
        print(f"Đang kiểm thử ảnh {i+1}/{len(test_images)}: {os.path.basename(img_path)}")
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Không thể đọc ảnh: {img_path}")
            continue
        
        try:
            start_time = time.time()
            similar_images = system.search_similar_faces(img)
            end_time = time.time()
            
            search_time = end_time - start_time
            total_time += search_time
            successful_searches += 1
            
            print(f"  - Thời gian tìm kiếm: {search_time:.4f} giây")
            print(f"  - Kết quả tốt nhất: khoảng cách {similar_images[0][0]:.4f}")
            
        except Exception as e:
            print(f"Lỗi khi kiểm thử ảnh {img_path}: {e}")
    
    if successful_searches > 0:
        avg_time = total_time / successful_searches
        print(f"\n=== KẾT QUẢ ĐÁNH GIÁ ===")
        print(f"Số ảnh kiểm thử thành công: {successful_searches}/{len(test_images)}")
        print(f"Thời gian tìm kiếm trung bình: {avg_time:.4f} giây")
        print(f"Tổng thời gian: {total_time:.4f} giây")
    else:
        print("Không có ảnh nào được kiểm thử thành công")


def main():
    """
    Hàm main - điểm vào chính của chương trình
    """
    print("=== HỆ THỐNG TÌM KIẾM ẢNH MẶT NGƯỜI CAO TUỔI ===")
    print("Sử dụng đặc trưng LBP và HOG với khoảng cách Euclidean")
    print()
    
    # Khởi tạo hệ thống
    system = ElderlyFaceSearchSystem(data_dir='elderly_faces_dataset')
    
    print("\nChọn chế độ hoạt động:")
    print("1. Demo với giao diện đồ họa")
    print("2. Đánh giá hiệu suất hệ thống")
    print("3. Cập nhật cơ sở dữ liệu")
    
    choice = input("Nhập lựa chọn của bạn (1/2/3): ").strip()
    
    if choice == '1':
        print("Khởi động giao diện đồ họa...")
        system.create_gui()
        
    elif choice == '2':
        test_dir = input("Nhập đường dẫn đến thư mục ảnh kiểm thử: ").strip()
        if os.path.exists(test_dir):
            evaluate_system_performance(system, test_dir)
        else:
            print(f"Không tìm thấy thư mục: {test_dir}")
            
    elif choice == '3':
        print("Cập nhật cơ sở dữ liệu...")
        system.load_database()
        print("Cập nhật hoàn tất!")
        
    else:
        print("Lựa chọn không hợp lệ. Khởi động giao diện đồ họa...")
        system.create_gui()


if __name__ == "__main__":
    main()