import sys
import os
import shutil
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QCheckBox, QTextEdit, QProgressBar, QListWidget, QListWidgetItem, QMessageBox, QSplitter,
    QGridLayout, QFrame, QScrollArea, QComboBox, QSpinBox, QGroupBox
)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer
from PIL import Image
import json
import os.path
from functools import lru_cache
from core.sorter import sort_images_by_blur
from utils.image_loader import load_images_from_folder
from core.face_filter import detect_face_attributes
from core.face_cluster import get_face_embedding, get_image_hash, are_images_duplicates
from core.analyzer import analyze_exposure, calculate_image_score
import qtmodern.styles
import qtmodern.windows

class ProcessingThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    log = pyqtSignal(str)
    
    def __init__(self, app):
        super().__init__()
        self.app = app
        
    def run(self):
        self.app.process_images()


class AilbumsApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ailbums Culling App")
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".ailbums_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.setGeometry(200, 100, 1200, 750)
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 4px;
            }
        """)
        self.folder_path = ""
        self.images = {}
        self.known_embeddings = []
        self.seen_hashes = []
        self.exported = 0
        self.image_status = {}
        self.image_scores = {}
        self.thumbnail_cache = {}
        self.filter_settings = {
            "min_score": 5,
            "sort_by": "score",
            "show_rejected": True
        }

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        
        # Create modern header
        header = QFrame()
        header.setStyleSheet("background-color: white; border-radius: 8px;")
        header_layout = QHBoxLayout(header)
        
        title = QLabel("Ailbums")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #1976D2;")
        header_layout.addWidget(title)
        
        self.folder_btn = QPushButton("Select Folder")
        self.folder_btn.clicked.connect(self.select_folder)
        header_layout.addWidget(self.folder_btn)
        
        main_layout.addWidget(header)
        # Add filter controls
        filter_box = QGroupBox("Filters & Sorting")
        filter_layout = QHBoxLayout()
        
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Score", "Blur", "Exposure"])
        self.sort_combo.currentTextChanged.connect(self.apply_filters)
        
        self.min_score = QSpinBox()
        self.min_score.setRange(0, 10)
        self.min_score.setValue(5)
        self.min_score.valueChanged.connect(self.apply_filters)
        
        filter_layout.addWidget(QLabel("Sort by:"))
        filter_layout.addWidget(self.sort_combo)
        filter_layout.addWidget(QLabel("Min score:"))
        filter_layout.addWidget(self.min_score)
        filter_box.setLayout(filter_layout)
        main_layout.addWidget(filter_box)

        # Create grid for thumbnails
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        content = QWidget()
        self.grid = QGridLayout(content)
        scroll.setWidget(content)
        
        main_layout.addWidget(scroll)

        options = QHBoxLayout()
        self.eyes_cb = QCheckBox("Filter closed eyes")
        self.eyes_cb.setChecked(True)
        self.smile_cb = QCheckBox("Filter no smile")
        self.smile_cb.setChecked(True)
        self.dup_cb = QCheckBox("Filter duplicates")
        self.dup_cb.setChecked(True)
        options.addWidget(self.eyes_cb)
        options.addWidget(self.smile_cb)
        options.addWidget(self.dup_cb)

        self.processing_thread = ProcessingThread(self)
        self.start_btn = QPushButton("Start Culling")
        self.start_btn.clicked.connect(self.run_culling)

        self.progress = QProgressBar()

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(100)

        main_layout.addLayout(options)
        
        # Add export options
        export_box = QGroupBox("Export Options")
        export_layout = QHBoxLayout()
        
        self.export_threshold = QSpinBox()
        self.export_threshold.setRange(0, 10)
        self.export_threshold.setValue(7)
        
        export_btn = QPushButton("Export Selected")
        export_btn.clicked.connect(self.export_selected)
        
        export_layout.addWidget(QLabel("Quality threshold:"))
        export_layout.addWidget(self.export_threshold)
        export_layout.addWidget(export_btn)
        export_box.setLayout(export_layout)
        main_layout.addWidget(export_box)
        main_layout.addWidget(self.start_btn)
        main_layout.addWidget(self.progress)
        layout.addWidget(self.log_box)
        layout.addWidget(QLabel("Approved & Rejected Thumbnails:"))
        layout.addWidget(self.thumb_list)

        self.setLayout(layout)

    def select_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select folder")
        if path:
            self.clear_cache()
            self.folder_path = path
            self.folder_label.setText(f"📁 {path}")
            self.log_box.append(f"Loaded folder: {path}")
            self.load_images()

    @lru_cache(maxsize=100)
    def get_cached_thumbnail(self, filename):
        cache_path = os.path.join(self.cache_dir, f"{filename}.json")
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None

    def save_to_cache(self, filename, data):
        cache_path = os.path.join(self.cache_dir, f"{filename}.json")
        with open(cache_path, 'w') as f:
            json.dump(data, f)

    def clear_cache(self):
        for file in os.listdir(self.cache_dir):
            os.remove(os.path.join(self.cache_dir, file))

    def apply_filters(self):
        sort_by = self.sort_combo.currentText().lower()
        min_score = self.min_score.value()
        
        filtered_images = {}
        for filename, scores in self.image_scores.items():
            if scores["total"] >= min_score:
                filtered_images[filename] = scores
        
        sorted_images = sorted(
            filtered_images.items(),
            key=lambda x: x[1][sort_by] if sort_by != "score" else x[1]["total"],
            reverse=True
        )
        
        self.update_grid(sorted_images)

    def update_grid(self, sorted_images):
        # Clear existing grid
        for i in reversed(range(self.grid.count())): 
            self.grid.itemAt(i).widget().setParent(None)
        
        # Add filtered and sorted images
        row = 0
        col = 0
        for filename, scores in sorted_images:
            thumb_widget = self.create_thumbnail_widget(filename, scores)
            self.grid.addWidget(thumb_widget, row, col)
            col += 1
            if col > 3:
                col = 0
                row += 1

    def create_thumbnail_widget(self, filename, scores):
        widget = QFrame()
        widget.setStyleSheet("background: white; border-radius: 8px; padding: 8px;")
        layout = QVBoxLayout(widget)
        
        # Image thumbnail
        img_label = QLabel()
        pixmap = self.get_thumbnail(filename)
        img_label.setPixmap(pixmap)
        layout.addWidget(img_label)
        
        # Stats
        stats = QLabel(f"""
            Score: {scores['total']:.1f}
            Blur: {scores['blur']:.0f}
            Eyes: {'✓' if scores['face']['eyes_open'] else '✗'}
            Smile: {'✓' if scores['face']['smiling'] else '✗'}
            Exposure: {scores['exposure']['quality']}
        """)
        stats.setStyleSheet("font-size: 10px;")
        layout.addWidget(stats)
        
        return widget
    def process_images(self):
        for i, (filename, img) in enumerate(self.images.items()):
            try:
                # Calculate all scores
                blur_score = sort_images_by_blur({filename: img})[filename]
                face_attrs = detect_face_attributes(img)
                exposure_data = analyze_exposure(img)
                
                # Calculate final score
                final_score = calculate_image_score(blur_score, face_attrs, exposure_data)
                
                self.image_scores[filename] = {
                    "total": final_score,
                    "blur": blur_score,
                    "face": face_attrs,
                    "exposure": exposure_data
                }
                
                self.processing_thread.progress.emit(i + 1)
                
            except Exception as e:
                self.processing_thread.log.emit(f"Error processing {filename}: {str(e)}")

    def load_images(self):
        self.thumb_list.clear()
        self.images = load_images_from_folder(self.folder_path)
        for filename, img in self.images.items():
            self.add_thumbnail(img, filename, "Pending")

    def run_culling(self):
        self.processing_thread.start()
        rejected_folder = os.path.join(self.folder_path, "Rejected")
        os.makedirs(approved_folder, exist_ok=True)
        os.makedirs(rejected_folder, exist_ok=True)

        sorted_results = sort_images_by_blur(self.images)
        sorted_items = sorted(sorted_results.items(), key=lambda x: x[1], reverse=True)

        self.progress.setMaximum(min(50, len(sorted_items)))
        self.exported = 0
        self.known_embeddings = []
        self.seen_hashes = []

        for i, (filename, _) in enumerate(sorted_items):
            if self.exported >= 50:
                break

            src_path = os.path.join(self.folder_path, filename)
            img = self.images[filename]
            reject = False
            reason = []

            try:
                attributes = None
                if self.smile_cb.isChecked() or self.eyes_cb.isChecked():
                    try:
                        attributes = detect_face_attributes(img)
                    except Exception:
                        reason.append("no face detected")

                if attributes:
                    if self.eyes_cb.isChecked() and not attributes.get("eyes_open"):
                        reason.append("eyes closed")
                        reject = True
                    if self.smile_cb.isChecked() and not attributes.get("smiling"):
                        reason.append("not smiling")
                        reject = True

                if not reject and self.dup_cb.isChecked():
                    img_hash = get_image_hash(img)
                    if any(are_images_duplicates(img_hash, h) for h in self.seen_hashes):
                        reason.append("duplicate")
                        reject = True
                    else:
                        self.seen_hashes.append(img_hash)

                if not reject:
                    try:
                        embedding = get_face_embedding(img)
                        if embedding is not None:
                            if any(np.linalg.norm(embedding - e) < 0.6 for e in self.known_embeddings):
                                reason.append("similar face")
                                reject = True
                            else:
                                self.known_embeddings.append(embedding)
                    except Exception:
                        reason.append("embedding error")

                dest = rejected_folder if reject else approved_folder
                shutil.copyfile(src_path, os.path.join(dest, filename))

                if not reject:
                    self.exported += 1
                    self.log_box.append(f"✅ {filename}")
                    self.image_status[filename] = "Approved"
                else:
                    self.log_box.append(f"❌ {filename}: {'; '.join(reason)}")
                    self.image_status[filename] = "Rejected"

                self.update_thumbnail_status(filename)

            except Exception as e:
                self.log_box.append(f"⚠️ {filename} failed: {e}")

            self.progress.setValue(self.exported)

        self.log_box.append(f"\n🎉 {self.exported} photos exported to {approved_folder}")

    def add_thumbnail(self, img, filename, status):
        try:
            pil_img = Image.fromarray(img[:, :, ::-1])
            pil_img.thumbnail((140, 140))
            qt_img = QImage(pil_img.tobytes(), pil_img.width, pil_img.height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img.rgbSwapped())
            icon = QIcon(pixmap)
            item = QListWidgetItem(icon, f"{filename} - {status}")
            item.setData(Qt.UserRole, (img, filename))
            self.thumb_list.addItem(item)
        except Exception as e:
            self.log_box.append(f"❌ Thumbnail failed for {filename}: {e}")

    def update_thumbnail_status(self, filename):
        for i in range(self.thumb_list.count()):
            item = self.thumb_list.item(i)
            img, fname = item.data(Qt.UserRole)
            if fname == filename:
                status = self.image_status.get(filename, "Pending")
                item.setText(f"{fname} - {status}")
                break

    def preview_full_image(self, item):
        img, fname = item.data(Qt.UserRole)
        win = QWidget()
        win.setWindowTitle(fname)
        layout = QVBoxLayout()
        label = QLabel()
        pixmap = QPixmap.fromImage(QImage(img[:, :, ::-1].tobytes(), img.shape[1], img.shape[0], QImage.Format_RGB888).rgbSwapped())
        label.setPixmap(pixmap.scaledToWidth(800, Qt.SmoothTransformation))
        layout.addWidget(label)
        win.setLayout(layout)
        win.resize(820, 600)
        win.show()

    def export_selected(self):
        threshold = self.export_threshold.value()
        export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        
        if export_dir:
            exported = 0
            for filename, scores in self.image_scores.items():
                if scores["total"] >= threshold:
                    src = os.path.join(self.folder_path, filename)
                    dst = os.path.join(export_dir, filename)
                    shutil.copy2(src, dst)
                    exported += 1
            
            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported {exported} images with score >= {threshold}"
            )

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AilbumsApp()
    window.show()
    sys.exit(app.exec_())