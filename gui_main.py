import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from PIL import Image, ImageTk
import threading
import os
from core.sorter import sort_images_by_blur
from utils.image_loader import load_images_from_folder
from core.face_filter import detect_face_attributes
from core.face_cluster import get_face_embedding, get_image_hash, are_images_duplicates
import shutil
import numpy as np

class AilbumsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ailbums Culling App")
        self.root.geometry("800x600")

        self.folder_path = tk.StringVar()

        # --- UI Layout ---
        top_frame = tk.Frame(root)
        top_frame.pack(pady=5)

        tk.Label(top_frame, text="Select image folder:").pack(side=tk.LEFT)
        tk.Entry(top_frame, textvariable=self.folder_path, width=60).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Browse", command=self.browse_folder).pack(side=tk.LEFT)

        tk.Button(root, text="Start Culling", command=self.run_culling_thread).pack(pady=5)

        self.progress = ttk.Progressbar(root, orient="horizontal", length=600, mode="determinate")
        self.progress.pack(pady=5)

        self.output_box = scrolledtext.ScrolledText(root, height=8, width=95)
        self.output_box.pack(pady=5)

        self.thumbnail_canvas = tk.Canvas(root, width=760, height=200, bg="white")
        self.thumbnail_canvas.pack(pady=10)
        self.thumbnail_row_y = 10
        self.thumbnails = []

    def browse_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.folder_path.set(path)

    def run_culling_thread(self):
        threading.Thread(target=self.run_culling).start()

    def run_culling(self):
        folder = self.folder_path.get()
        if not os.path.isdir(folder):
            messagebox.showerror("Error", "Please select a valid folder.")
            return

        self.output_box.delete("1.0", tk.END)
        self.thumbnail_canvas.delete("all")
        self.thumbnails.clear()
        self.output_box.insert(tk.END, f"📂 Loading images from: {folder}\n")

        images = load_images_from_folder(folder)
        results = sort_images_by_blur(images)
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

        approved_folder = os.path.join(folder, "Approved")
        rejected_folder = os.path.join(folder, "Rejected")
        os.makedirs(approved_folder, exist_ok=True)
        os.makedirs(rejected_folder, exist_ok=True)

        export_count = 50
        seen_hashes = []
        known_embeddings = []
        exported = 0
        total = min(export_count, len(sorted_results))
        self.progress["maximum"] = total

        x, y = 10, self.thumbnail_row_y

        for i, (filename, _) in enumerate(sorted_results):
            if exported >= export_count:
                break

            src_path = os.path.join(folder, filename)
            img = images[filename]

            try:
                attributes = detect_face_attributes(img)
                if not (attributes["eyes_open"] and attributes["smiling"]):
                    shutil.copyfile(src_path, os.path.join(rejected_folder, filename))
                    continue

                img_hash = get_image_hash(img)
                if any(are_images_duplicates(img_hash, h) for h in seen_hashes):
                    shutil.copyfile(src_path, os.path.join(rejected_folder, filename))
                    continue
                seen_hashes.append(img_hash)

                embedding = get_face_embedding(img)
                if embedding is not None:
                    if any(np.linalg.norm(embedding - e) < 0.6 for e in known_embeddings):
                        shutil.copyfile(src_path, os.path.join(rejected_folder, filename))
                        continue
                    known_embeddings.append(embedding)

                # APPROVED
                shutil.copyfile(src_path, os.path.join(approved_folder, filename))
                exported += 1
                self.output_box.insert(tk.END, f"✅ {filename}\n")

                # Thumbnail
                pil_img = Image.fromarray(img[:, :, ::-1])
                pil_img.thumbnail((100, 100))
                thumb = ImageTk.PhotoImage(pil_img)
                self.thumbnails.append(thumb)  # Keep ref alive
                self.thumbnail_canvas.create_image(x, y, anchor=tk.NW, image=thumb)
                x += 110
                if x > 700:
                    x = 10
                    y += 110

                self.progress["value"] = exported
                self.root.update_idletasks()

            except Exception as e:
                self.output_box.insert(tk.END, f"❌ {filename}: {e}\n")
                shutil.copyfile(src_path, os.path.join(rejected_folder, filename))

        self.output_box.insert(tk.END, f"\n🎉 {exported} photos exported to {approved_folder}\n")

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = AilbumsApp(root)
    root.mainloop()
