import sys
import os
import shutil
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QCheckBox, QTextEdit, QProgressBar, QListWidget, QListWidgetItem, QMessageBox, QScrollArea
)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QSize
from PIL import Image
from core.sorter import sort_images_by_blur
from utils.image_loader import load_images_from_folder
from core.face_filter import detect_face_attributes
from core.face_cluster import get_face_embedding, get_image_hash, are_images_duplicates


class AilbumsApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ailbums Culling App")
        self.setGeometry(200, 100, 1000, 700)
        self.folder_path = ""
        self.images = {}
        self.thumbnails = []
        self.known_embeddings = []
        self.seen_hashes = []
        self.exported = 0

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Top bar
        top_bar = QHBoxLayout()
        self.folder_label = QLabel("Select image folder:")
        self.folder_btn = QPushButton("Browse")
        self.folder_btn.clicked.connect(self.select_folder)
        top_bar.addWidget(self.folder_label)
        top_bar.addWidget(self.folder_btn)

        # Checkboxes
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

        # Start button
        self.start_btn = QPushButton("Start Culling")
        self.start_btn.clicked.connect(self.run_culling)

        # Output box
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        # Progress
        self.progress = QProgressBar()

        # Thumbnails
        self.thumb_list = QListWidget()
        self.thumb_list.setIconSize(QSize(100, 100))

        layout.addLayout(top_bar)
        layout.addLayout(options)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.progress)
        layout.addWidget(self.log_box)
        layout.addWidget(QLabel("Approved Thumbnails:"))
        layout.addWidget(self.thumb_list)

        self.setLayout(layout)

    def select_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select folder")
        if path:
            self.folder_path = path
            self.folder_label.setText(f"📁 {path}")
            self.log_box.append(f"Loaded folder: {path}")

    def run_culling(self):
        if not self.folder_path:
            QMessageBox.warning(self, "Error", "No folder selected.")
            return

        approved_folder = os.path.join(self.folder_path, "Approved")
        rejected_folder = os.path.join(self.folder_path, "Rejected")
        os.makedirs(approved_folder, exist_ok=True)
        os.makedirs(rejected_folder, exist_ok=True)

        self.images = load_images_from_folder(self.folder_path)
        sorted_results = sort_images_by_blur(self.images)
        sorted_items = sorted(sorted_results.items(), key=lambda x: x[1], reverse=True)

        self.progress.setMaximum(min(50, len(sorted_items)))
        self.exported = 0

        for i, (filename, _) in enumerate(sorted_items):
            if self.exported >= 50:
                break

            src_path = os.path.join(self.folder_path, filename)
            img = self.images[filename]
            reject = False

            try:
                if self.smile_cb.isChecked() or self.eyes_cb.isChecked():
                    attributes = detect_face_attributes(img)
                    if self.eyes_cb.isChecked() and attributes.get("eyes_open") is False:
                        self.log_box.append(f"❌ {filename}: eyes closed")
                        reject = True
                    if self.smile_cb.isChecked() and not attributes.get("smiling", False):
                        self.log_box.append(f"❌ {filename}: not smiling")
                        reject = True

                if not reject and self.dup_cb.isChecked():
                    img_hash = get_image_hash(img)
                    if any(are_images_duplicates(img_hash, h) for h in self.seen_hashes):
                        self.log_box.append(f"❌ {filename}: duplicate detected")
                        reject = True
                    else:
                        self.seen_hashes.append(img_hash)

                if not reject:
                    embedding = get_face_embedding(img)
                    if embedding is not None:
                        if any(np.linalg.norm(embedding - e) < 0.6 for e in self.known_embeddings):
                            self.log_box.append(f"❌ {filename}: same person")
                            reject = True
                        else:
                            self.known_embeddings.append(embedding)

                dest = rejected_folder if reject else approved_folder
                shutil.copyfile(src_path, os.path.join(dest, filename))

                if not reject:
                    self.exported += 1
                    self.log_box.append(f"✅ {filename}")
                    self.add_thumbnail(img, filename)

            except Exception as e:
                self.log_box.append(f"⚠️ {filename} failed: {e}")

            self.progress.setValue(self.exported)

        self.log_box.append(f"\n🎉 {self.exported} photos exported to {approved_folder}")

    def add_thumbnail(self, img, filename):
        try:
            pil_img = Image.fromarray(img[:, :, ::-1])
            pil_img.thumbnail((100, 100))
            qt_img = QImage(pil_img.tobytes(), pil_img.width, pil_img.height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img.rgbSwapped())
            item = QListWidgetItem(QIcon(pixmap), filename)
            self.thumb_list.addItem(item)
        except Exception as e:
            self.log_box.append(f"❌ Thumbnail failed for {filename}: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AilbumsApp()
    window.show()
    sys.exit(app.exec_())




