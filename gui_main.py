import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from core.sorter import sort_images_by_blur
from utils.image_loader import load_images_from_folder
from core.face_filter import detect_face_attributes
from core.face_cluster import get_face_embedding, get_image_hash, are_images_duplicates

class CullingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ailbums Culling App")
        self.geometry("800x600")

        # Folder selection
        self.folder_path = tk.StringVar()
        folder_frame = tk.Frame(self)
        folder_frame.pack(pady=10)
        tk.Label(folder_frame, text="Image Folder:").pack(side=tk.LEFT)
        tk.Entry(folder_frame, textvariable=self.folder_path, width=60).pack(side=tk.LEFT, padx=5)
        tk.Button(folder_frame, text="Browse", command=self.browse_folder).pack(side=tk.LEFT)

        # Start button
        tk.Button(self, text="Start Culling", command=self.start_culling).pack(pady=10)

        # Output log
        self.log = scrolledtext.ScrolledText(self, wrap=tk.WORD)
        self.log.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def browse_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.folder_path.set(path)

    def log_message(self, msg):
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)

    def start_culling(self):
        folder = self.folder_path.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Please select a valid image folder.")
            return

        # Run culling in a separate thread to keep UI responsive
        thread = threading.Thread(target=self.run_culling, args=(folder,))
        thread.start()

    def run_culling(self, folder):
        self.log_message("Loading images...")
        images = load_images_from_folder(folder)

        self.log_message("Sorting by sharpness...")
        results = sort_images_by_blur(images)
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

        # Setup export folders
        approved_folder = os.path.join(folder, "Approved")
        rejected_folder = os.path.join(folder, "Rejected")
        os.makedirs(approved_folder, exist_ok=True)
        os.makedirs(rejected_folder, exist_ok=True)

        self.log_message("Analyzing faces and filtering...")
        exported = 0
        seen_hashes = []
        known_embeddings = []
        export_count = 50

        for filename, _ in sorted_results:
            if exported >= export_count:
                break
            src_path = os.path.join(folder, filename)
            img = images[filename]
            try:
                attrs = detect_face_attributes(img)
                if not (attrs['eyes_open'] and attrs['smiling']):
                    dst = os.path.join(rejected_folder, filename)
                else:
                    img_hash = get_image_hash(img)
                    if any(are_images_duplicates(img_hash, h) for h in seen_hashes):
                        dst = os.path.join(rejected_folder, filename)
                    else:
                        seen_hashes.append(img_hash)
                        emb = get_face_embedding(img)
                        if emb is not None and any((emb - e).dot(emb - e) < 0.36 for e in known_embeddings):
                            dst = os.path.join(rejected_folder, filename)
                        else:
                            if emb is not None:
                                known_embeddings.append(emb)
                            dst = os.path.join(approved_folder, filename)
                            exported += 1
                shutil.copyfile(src_path, dst)
                self.log_message(f"Processed: {filename}")
            except Exception as e:
                shutil.copyfile(src_path, os.path.join(rejected_folder, filename))
                self.log_message(f"Error processing {filename}: {e}")

        self.log_message(f"Completed. Exported {exported} photos to {approved_folder}")

if __name__ == '__main__':
    app = CullingApp()
    app.mainloop()
