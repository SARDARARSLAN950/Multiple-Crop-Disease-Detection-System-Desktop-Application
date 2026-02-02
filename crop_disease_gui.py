import sys
import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QComboBox, QTextEdit, QGroupBox, QScrollArea, QListWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QColor, QPen, QPolygonF
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load YOLO models
try:
    cotton_model = YOLO("cotton leaf diseas 100.pt")
    wheat_model = YOLO("rice yolov8s.pt")
    rice_model = YOLO("wheat disese best.pt")
    logger.info("All YOLO models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load YOLO models: {e}")
    raise

# Define classes
COTTON_CLASSES = ['Bacterial Blight', 'Curl virus', 'Fussarium wilt', 'Healthy']
WHEAT_CLASSES = ["Brown_Rust", "Healthy", "Loose_Smut", "Septoria", "Yellow_Rust"]
RICE_CLASSES = ['Bacterial_Leaf_Blight', 'Brown_Spot', 'HealthyLeaf', 'Leaf_Blast', 
                'Leaf_Scald', 'Narrow_Brown_Leaf_Spot', 'Neck_Blast', 'Rice_Hispa']

# Define color maps
COTTON_COLOR_MAP = {
    "Bacterial Blight": sv.Color(255, 0, 0),
    "Curl virus": sv.Color(0, 255, 0),
    "Fussarium wilt": sv.Color(0, 0, 255),
    "Healthy": sv.Color(255, 255, 0)
}
WHEAT_COLOR_MAP = {
    "brown_rust": sv.Color(165, 42, 42),
    "healthy": sv.Color(255, 255, 0),
    "loose_smut": sv.Color(128, 128, 128),
    "septoria": sv.Color(0, 128, 0),
    "yellow_rust": sv.Color(255, 215, 0)
}
RICE_COLOR_MAP = {
    "Bacterial_Leaf_Blight": sv.Color(255, 0, 0),
    "Brown_Spot": sv.Color(0, 0, 255),
    "HealthyLeaf": sv.Color(255, 255, 0),
    "Leaf_Blast": sv.Color(0, 255, 0),
    "Leaf_Scald": sv.Color(255, 165, 0),
    "Narrow_Brown_Leaf_Spot": sv.Color(128, 0, 128),
    "Neck_Blast": sv.Color(255, 0, 255),
    "Rice_Hispa": sv.Color(0, 255, 255)
}

# Define solutions and products
COTTON_SOLUTIONS = {
    "bacterial blight": "Use copper-based bactericides and practice crop rotation.",
    "curl virus": "Control whitefly vectors and use resistant varieties.",
    "fusarium wilt": "Use resistant varieties and soil solarization.",
    "healthy": "No treatment needed."
}
COTTON_PRODUCTS = {
    "bacterial blight": ["Copper Fungicide A", "Bactericide B"],
    "curl virus": ["Insecticide C", "Resistant Seed D"],
    "fusarium wilt": ["Soil Treatment E", "Fungicide F"],
    "healthy": []
}

WHEAT_SOLUTIONS = {
    "brown_rust": "Apply appropriate fungicides and remove infected plant debris.",
    "healthy": "No treatment needed.",
    "loose_smut": "Use certified disease-free seeds and crop rotation.",
    "septoria": "Apply fungicides and practice crop rotation.",
    "yellow_rust": "Use resistant varieties and fungicide application."
}
WHEAT_PRODUCTS = {
    "brown_rust": ["Fungicide Alpha", "Plant Debris Remover"],
    "healthy": ["No treatment needed"],
    "loose_smut": ["Certified Seeds", "Crop Rotation Guide"],
    "septoria": ["Fungicide Beta", "Crop Rotation Guide"],
    "yellow_rust": ["Resistant Seed Variety", "Fungicide Gamma"]
}

RICE_SOLUTIONS = {
    "bacterial_leaf_blight": "Apply fungicides and remove infected plant debris.",
    "brown_spot": "Use resistant varieties and proper fertilization.",
    "healthyleaf": "No treatment needed.",
    "leaf_blast": "Use resistant varieties and balanced fertilization.",
    "leaf_scald": "Use fungicides and maintain proper water management.",
    "narrow_brown_leaf_spot": "Apply fungicides during early flowering stage.",
    "neck_blast": "Improve drainage and apply fungicides.",
    "rice_hispa": "Control vector insects and use resistant varieties."
}
RICE_PRODUCTS = {
    "bacterial_leaf_blight": ["Fungicide A", "Fungicide B"],
    "brown_spot": ["Fertilizer C", "Resistant Seed D"],
    "healthyleaf": [],
    "leaf_blast": ["Fungicide H", "Balanced Fertilizer"],
    "leaf_scald": ["Fungicide G", "Water Management Tools"],
    "narrow_brown_leaf_spot": ["Fungicide E", "Fungicide F"],
    "neck_blast": ["Drainage Improvement Tools", "Fungicide I"],
    "rice_hispa": ["Insecticide J", "Resistant Seed K"]
}


class ProcessingThread(QThread):
    """Thread for processing images without blocking the GUI"""
    finished = pyqtSignal(object, str, dict, list)
    
    def __init__(self, image_path, model, class_names, color_map):
        super().__init__()
        self.image_path = image_path
        self.model = model
        self.class_names = class_names
        self.color_map = color_map
    
    def run(self):
        try:
            # Load image
            image = cv2.imread(self.image_path)
            if image is None:
                self.finished.emit(None, "Error: Could not load image", {}, [])
                return

            # Resize image
            target_width, target_height = 1280, 720
            image = cv2.resize(image, (target_width, target_height))

            # Run YOLO detection
            results = self.model(image)[0]
            detections = sv.Detections.from_ultralytics(results)

            # Count diseases
            disease_counts = {}
            detected_areas = []
            
            if len(detections) > 0:
                for idx, class_id in enumerate(detections.class_id):
                    class_name = self.class_names[class_id]
                    disease_counts[class_name] = disease_counts.get(class_name, 0) + 1
                    
                    # Calculate area of detection
                    box = detections.xyxy[idx]
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    area = int(width * height)
                    detected_areas.append({
                        'disease': class_name,
                        'confidence': float(detections.confidence[idx]),
                        'area': area,
                        'bbox': box
                    })

            # Assign colors and annotate
            annotated_image = image.copy()
            if len(detections) > 0:
                for detection_idx, xyxy in enumerate(detections.xyxy):
                    class_name = self.class_names[detections.class_id[detection_idx]]
                    color = self.color_map.get(class_name, sv.Color(128, 128, 128))
                    
                    single_detection = sv.Detections(
                        xyxy=np.array([xyxy]),
                        class_id=np.array([detections.class_id[detection_idx]]),
                        confidence=np.array([detections.confidence[detection_idx]])
                    )

                    box_annotator = sv.BoxAnnotator(color=color)
                    label_annotator = sv.LabelAnnotator(
                        color=color,
                        text_color=sv.Color(255, 255, 255),
                        text_position=sv.Position.TOP_LEFT
                    )

                    annotated_image = box_annotator.annotate(
                        scene=annotated_image,
                        detections=single_detection
                    )

                    labels = [f"{class_name}: {single_detection.confidence[0]:.2f}"]
                    annotated_image = label_annotator.annotate(
                        scene=annotated_image,
                        detections=single_detection,
                        labels=labels
                    )

            # Add disease count overlay
            y_offset = 30
            for disease, count in disease_counts.items():
                text = f"{disease}: {count}"
                cv2.putText(
                    annotated_image, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
                y_offset += 30

            self.finished.emit(annotated_image, None, disease_counts, detected_areas)

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            self.finished.emit(None, f"Error: {str(e)}", {}, [])


class CropDiseaseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.processed_image = None
        self.init_ui()
    
    def create_crop_icons_pixmap(self, size=60):
        """Create a pixmap with rice, wheat, and cotton icons"""
        pixmap = QPixmap(size * 3 + 40, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Rice icon (grain/seed)
        painter.setBrush(QColor(255, 223, 186))
        painter.setPen(QPen(QColor(139, 69, 19), 2))
        for i in range(3):
            painter.drawEllipse(10 + i*8, 15 + i*10, 12, 20)
        
        # Wheat icon (wheat stalk)
        painter.setPen(QPen(QColor(218, 165, 32), 3))
        painter.drawLine(size + 30, size - 5, size + 30, 15)
        painter.setBrush(QColor(255, 215, 0))
        for i in range(5):
            painter.drawEllipse(size + 20 + (i % 2) * 10, 15 + i * 8, 8, 12)
        
        # Cotton icon (cotton boll)
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(QPen(QColor(34, 139, 34), 2))
        painter.drawEllipse(size * 2 + 25, 20, 25, 25)
        painter.setBrush(QColor(240, 240, 240))
        for i in range(4):
            angle = i * 90
            x = size * 2 + 37 + 12 * np.cos(np.radians(angle))
            y = 32 + 12 * np.sin(np.radians(angle))
            painter.drawEllipse(int(x), int(y), 10, 10)
        
        painter.end()
        return pixmap
    
    def init_ui(self):
        self.setWindowTitle('BUETK Crop Disease Detection System')
        self.setGeometry(50, 50, 1600, 950)
        
        # Set modern style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        
        # Header Section
        header_layout = QHBoxLayout()
        header_widget = QWidget()
        header_widget.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #1e3c72, stop:1 #2a5298);
            border-radius: 10px;
            padding: 10px;
        """)
        header_container = QHBoxLayout()
        header_widget.setLayout(header_container)
        
        # University Logo (Left)
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)
        # Create BUETK logo placeholder
        logo_pixmap = QPixmap(100, 100)
        logo_pixmap.fill(Qt.transparent)
        painter = QPainter(logo_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw circular background
        painter.setBrush(QColor(255, 255, 255))
        painter.setPen(QPen(QColor(0, 0, 0), 3))
        painter.drawEllipse(5, 5, 90, 90)
        
        # Draw gear (simplified)
        painter.setBrush(QColor(255, 193, 7))
        painter.setPen(QPen(QColor(255, 193, 7), 2))
        for i in range(8):
            angle = i * 45
            x1 = 50 + 35 * np.cos(np.radians(angle))
            y1 = 50 + 35 * np.sin(np.radians(angle))
            painter.drawRect(int(x1-3), int(y1-3), 6, 6)
        painter.drawEllipse(30, 30, 40, 40)
        
        # Draw book and building (simplified)
        painter.setBrush(QColor(139, 69, 19))
        painter.drawRect(35, 50, 12, 18)
        painter.setBrush(QColor(205, 133, 63))
        painter.drawPolygon(QPolygonF([
            QPointF(60, 60), QPointF(70, 50), QPointF(70, 68)
        ]))
        
        # Draw text on logo
        painter.setPen(QColor(0, 0, 0))
        painter.setFont(QFont('Arial', 8, QFont.Bold))
        painter.drawText(25, 85, 'BUETK')
        
        painter.end()
        logo_label.setPixmap(logo_pixmap)
        header_container.addWidget(logo_label)
        
        # Title Section (Center)
        title_container = QVBoxLayout()
        title_label = QLabel('BUETK CROP DISEASE DETECTION')
        title_label.setFont(QFont('Arial', 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet('color: white; padding: 5px;')
        
        subtitle_label = QLabel('Balochistan University of Engineering & Technology, Khuzdar')
        subtitle_label.setFont(QFont('Arial', 12))
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet('color: #e0e0e0; padding: 2px;')
        
        title_container.addWidget(title_label)
        title_container.addWidget(subtitle_label)
        header_container.addLayout(title_container)
        
        # Crop Icons (Right)
        crop_icons_label = QLabel()
        crop_icons_label.setAlignment(Qt.AlignCenter)
        crop_icons_pixmap = self.create_crop_icons_pixmap()
        crop_icons_label.setPixmap(crop_icons_pixmap)
        header_container.addWidget(crop_icons_label)
        
        main_layout.addWidget(header_widget)
        
        # Content Section
        content_layout = QHBoxLayout()
        
        # Left panel - Controls and input
        left_panel = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setMaximumWidth(450)
        left_widget.setLayout(left_panel)
        
        # Crop selection
        crop_group = QGroupBox('🌾 Select Crop Type')
        crop_group.setFont(QFont('Arial', 12, QFont.Bold))
        crop_layout = QVBoxLayout()
        self.crop_combo = QComboBox()
        self.crop_combo.addItems(['Select a crop', '🌾 Rice', '🌾 Wheat', '☁️ Cotton'])
        self.crop_combo.setFont(QFont('Arial', 11))
        self.crop_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 2px solid #4CAF50;
                border-radius: 5px;
                background-color: white;
            }
        """)
        crop_layout.addWidget(self.crop_combo)
        crop_group.setLayout(crop_layout)
        left_panel.addWidget(crop_group)
        
        # Image selection
        self.upload_btn = QPushButton('📁 Upload Image')
        self.upload_btn.clicked.connect(self.upload_image)
        self.upload_btn.setFont(QFont('Arial', 12, QFont.Bold))
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 12px;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        left_panel.addWidget(self.upload_btn)
        
        # Original image display
        original_group = QGroupBox('📷 Original Image')
        original_group.setFont(QFont('Arial', 12, QFont.Bold))
        original_layout = QVBoxLayout()
        self.original_image_label = QLabel()
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumSize(400, 350)
        self.original_image_label.setStyleSheet("""
            border: 3px dashed #cccccc;
            background-color: #fafafa;
            border-radius: 8px;
        """)
        self.original_image_label.setText('No image selected\n\nClick "Upload Image" to begin')
        self.original_image_label.setFont(QFont('Arial', 11))
        original_layout.addWidget(self.original_image_label)
        original_group.setLayout(original_layout)
        left_panel.addWidget(original_group)
        
        # Process button
        self.process_btn = QPushButton('🔍 Detect Disease')
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setEnabled(False)
        self.process_btn.setFont(QFont('Arial', 12, QFont.Bold))
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 12px;
                border-radius: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        left_panel.addWidget(self.process_btn)
        
        # Status label
        self.status_label = QLabel('Ready to detect diseases')
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont('Arial', 10))
        self.status_label.setStyleSheet("""
            color: #666;
            background-color: #e8f5e9;
            padding: 8px;
            border-radius: 5px;
            font-style: italic;
        """)
        left_panel.addWidget(self.status_label)
        
        left_panel.addStretch()
        content_layout.addWidget(left_widget)
        
        # Right panel - Results
        right_panel = QVBoxLayout()
        
        # Processed image display
        processed_group = QGroupBox('✅ Detection Results')
        processed_group.setFont(QFont('Arial', 12, QFont.Bold))
        processed_layout = QVBoxLayout()
        self.processed_image_label = QLabel()
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_label.setMinimumSize(700, 450)
        self.processed_image_label.setStyleSheet("""
            border: 3px solid #2196F3;
            background-color: #fafafa;
            border-radius: 8px;
        """)
        self.processed_image_label.setText('Detection results will appear here')
        self.processed_image_label.setFont(QFont('Arial', 11))
        processed_layout.addWidget(self.processed_image_label)
        processed_group.setLayout(processed_layout)
        right_panel.addWidget(processed_group)
        
        # Results tabs
        results_layout = QHBoxLayout()
        
        # Disease summary
        summary_group = QGroupBox('📊 Disease Summary')
        summary_group.setFont(QFont('Arial', 11, QFont.Bold))
        summary_layout = QVBoxLayout()
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(150)
        self.summary_text.setFont(QFont('Arial', 10))
        summary_layout.addWidget(self.summary_text)
        summary_group.setLayout(summary_layout)
        results_layout.addWidget(summary_group)
        
        # Area details
        area_group = QGroupBox('📏 Detection Areas')
        area_group.setFont(QFont('Arial', 11, QFont.Bold))
        area_layout = QVBoxLayout()
        self.area_list = QListWidget()
        self.area_list.setMaximumHeight(150)
        self.area_list.setFont(QFont('Arial', 10))
        area_layout.addWidget(self.area_list)
        area_group.setLayout(area_layout)
        results_layout.addWidget(area_group)
        
        right_panel.addLayout(results_layout)
        
        # Solutions and Products in one row
        bottom_results_layout = QHBoxLayout()
        
        # Solutions
        solution_group = QGroupBox('💊 Recommended Solutions')
        solution_group.setFont(QFont('Arial', 11, QFont.Bold))
        solution_layout = QVBoxLayout()
        self.solution_text = QTextEdit()
        self.solution_text.setReadOnly(True)
        self.solution_text.setFont(QFont('Arial', 10))
        solution_layout.addWidget(self.solution_text)
        solution_group.setLayout(solution_layout)
        bottom_results_layout.addWidget(solution_group)
        
        # Products
        product_group = QGroupBox('🛒 Recommended Products')
        product_group.setFont(QFont('Arial', 11, QFont.Bold))
        product_layout = QVBoxLayout()
        self.product_text = QTextEdit()
        self.product_text.setReadOnly(True)
        self.product_text.setFont(QFont('Arial', 10))
        product_layout.addWidget(self.product_text)
        product_group.setLayout(product_layout)
        bottom_results_layout.addWidget(product_group)
        
        right_panel.addLayout(bottom_results_layout)
        
        content_layout.addLayout(right_panel, 2)
        main_layout.addLayout(content_layout)
    
    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, 'Select Image', '', 
            'Image Files (*.png *.jpg *.jpeg)'
        )
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            scaled_pixmap = pixmap.scaled(
                400, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.original_image_label.setPixmap(scaled_pixmap)
            self.process_btn.setEnabled(True)
            self.status_label.setText('✓ Image loaded successfully - Ready to detect')
            self.status_label.setStyleSheet("""
                color: #2e7d32;
                background-color: #e8f5e9;
                padding: 8px;
                border-radius: 5px;
                font-weight: bold;
            """)
    
    def process_image(self):
        if not self.image_path:
            self.status_label.setText('⚠ Please upload an image first')
            return
        
        crop = self.crop_combo.currentText()
        if crop == 'Select a crop':
            self.status_label.setText('⚠ Please select a crop type')
            self.status_label.setStyleSheet("""
                color: #d32f2f;
                background-color: #ffebee;
                padding: 8px;
                border-radius: 5px;
                font-weight: bold;
            """)
            return
        
        # Remove emoji from crop name
        crop = crop.split(' ')[-1]
        
        # Select model and parameters based on crop
        if crop == 'Cotton':
            model = cotton_model
            class_names = COTTON_CLASSES
            color_map = COTTON_COLOR_MAP
            self.solutions_dict = COTTON_SOLUTIONS
            self.products_dict = COTTON_PRODUCTS
        elif crop == 'Wheat':
            model = wheat_model
            class_names = WHEAT_CLASSES
            color_map = WHEAT_COLOR_MAP
            self.solutions_dict = WHEAT_SOLUTIONS
            self.products_dict = WHEAT_PRODUCTS
        else:  # Rice
            model = rice_model
            class_names = RICE_CLASSES
            color_map = RICE_COLOR_MAP
            self.solutions_dict = RICE_SOLUTIONS
            self.products_dict = RICE_PRODUCTS
        
        self.status_label.setText('⏳ Processing image... Please wait')
        self.status_label.setStyleSheet("""
            color: #f57c00;
            background-color: #fff3e0;
            padding: 8px;
            border-radius: 5px;
            font-weight: bold;
        """)
        self.process_btn.setEnabled(False)
        
        # Start processing thread
        self.thread = ProcessingThread(self.image_path, model, class_names, color_map)
        self.thread.finished.connect(self.display_results)
        self.thread.start()
    
    def display_results(self, annotated_image, error, disease_counts, detected_areas):
        self.process_btn.setEnabled(True)
        
        if error:
            self.status_label.setText(f'❌ {error}')
            self.status_label.setStyleSheet("""
                color: #d32f2f;
                background-color: #ffebee;
                padding: 8px;
                border-radius: 5px;
                font-weight: bold;
            """)
            return
        
        # Display processed image
        if annotated_image is not None:
            height, width, channel = annotated_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(
                annotated_image.data, width, height, 
                bytes_per_line, QImage.Format_RGB888
            ).rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                700, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.processed_image_label.setPixmap(scaled_pixmap)
        
        # Display disease summary
        summary_html = '<h3 style="color: #1976d2;">Disease Detection Summary</h3>'
        if disease_counts:
            for disease, count in disease_counts.items():
                color = '#f44336' if disease.lower() != 'healthy' and 'healthy' not in disease.lower() else '#4caf50'
                summary_html += f'<p style="color: {color};"><b>{disease}:</b> {count} detection(s)</p>'
        else:
            summary_html += '<p style="color: #4caf50;">✓ No diseases detected - Crop appears healthy!</p>'
        self.summary_text.setHtml(summary_html)
        
        # Display area details
        self.area_list.clear()
        total_area = 0
        for idx, area_info in enumerate(detected_areas):
            area_text = f"#{idx+1} {area_info['disease']} - Area: {area_info['area']:,}px² - Confidence: {area_info['confidence']:.1%}"
            self.area_list.addItem(area_text)
            total_area += area_info['area']
        
        if detected_areas:
            self.area_list.addItem("")
            self.area_list.addItem(f"📊 Total Affected Area: {total_area:,} px²")
        
        # Display solutions
        solution_html = '<h3 style="color: #1976d2;">Treatment Solutions</h3>'
        for disease in disease_counts.keys():
            solution = self.solutions_dict.get(disease.lower().replace(' ', '_'), 
                                              "No solution available.")
            solution_html += f'<p><b style="color: #f57c00;">{disease}:</b> {solution}</p>'
        self.solution_text.setHtml(solution_html)
        
        # Display products
        product_html = '<h3 style="color: #1976d2;">Recommended Products</h3>'
        for disease in disease_counts.keys():
            products = self.products_dict.get(disease.lower().replace(' ', '_'), [])
            product_html += f'<p><b style="color: #f57c00;">{disease}:</b></p><ul>'
            if products:
                for product in products:
                    product_html += f'<li>{product}</li>'
            else:
                product_html += '<li style="color: #4caf50;">No products needed</li>'
            product_html += '</ul>'
        self.product_text.setHtml(product_html)
        
        self.status_label.setText('✓ Processing complete! Results displayed below.')
        self.status_label.setStyleSheet("""
            color: #2e7d32;
            background-color: #e8f5e9;
            padding: 8px;
            border-radius: 5px;
            font-weight: bold;
        """)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application icon (you can add your logo file here)
    # app.setWindowIcon(QIcon('path_to_logo.png'))
    
    window = CropDiseaseGUI()
    window.show()
    sys.exit(app.exec_())