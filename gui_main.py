import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk, Toplevel
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
        self.root.geometry("800x680")

        self.folder_path = tk.StringVar()
        self.apply_smile_filter = tk.BooleanVar(value=True)
        self.apply_eyes_filter = tk.BooleanVar(value=True)
        self.apply_duplicate_filter = tk.BooleanVar(value=True)

        top_frame = tk.Frame(root)
        top_frame.pack(pady=5)

        tk.Label(top_frame, text="Select image folder:").pack(side=tk.LEFT)
        tk.Entry(top_frame, textvariable=self.folder_path, width=60).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Browse", command=self.browse_folder).pack(side=tk.LEFT)

        options_frame = tk.Frame(root)
        options_frame.pack()

        tk.Checkbutton(options_frame, text="Filter closed eyes", variable=self.apply_eyes_filter).pack(side=tk.LEFT, padx=10)
        tk.Checkbutton(options_frame, text="Filter no smile", variable=self.apply_smile_filter).pack(side=tk.LEFT, padx=10)
        tk.Checkbutton(options_frame, text="Filter duplicates", variable=self.apply_duplicate_filter).pack(side=tk.LEFT, padx=10)

        tk.Button(root, text="Start Culling", command=self.run_culling_thread).pack(pady=5)

        self.progress = ttk.Progressbar(root, orient="horizontal", length=600, mode="determinate")
        self.progress.pack(pady=5)

        self.output_box = scrolledtext.ScrolledText(root, height=8, width=95)
        self.output_box.pack(pady=5)

        self.thumbnail_canvas = tk.Canvas(root, width=760, height=200, bg="white")
        self.thumbnail_canvas.pack(pady=10)
        self.thumbnail_row_y = 10
        self.thumbnails = []
        self.thumb_map = {}

    def browse_folder(self):
        path = filedialog.askdirectory(title="Select folder with images")
        if path:
            image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
            all_files = os.listdir(path)
            image_files = [f for f in all_files if f.lower().endswith(image_extensions)]

            if not image_files:
                messagebox.showwarning("No Images Found", "This folder contains no supported image files.")
                return

            self.folder_path.set(path)
            messagebox.showinfo("Images Found", f"📸 Found {len(image_files)} image(s) in the folder.")

    def run_culling_thread(self):
        threading.Thread(target=self.run_culling).start()

    def show_full_image(self, img_array, filename):
        top = Toplevel(self.root)
        top.title(filename)
        pil_img = Image.fromarray(img_array[:, :, ::-1])
        img = ImageTk.PhotoImage(pil_img)
        label = tk.Label(top, image=img)
        label.image = img
        label.pack()

    def run_culling(self):
        folder = self.folder_path.get()
        if not os.path.isdir(folder):
            messagebox.showerror("Error", "Please select a valid folder.")
            return

        self.output_box.delete("1.0", tk.END)
        self.thumbnail_canvas.delete("all")
        self.thumbnails.clear()
        self.thumb_map.clear()
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
                # Face filters
                if self.apply_smile_filter.get() or self.apply_eyes_filter.get():
                    try:
                        attributes = detect_face_attributes(img)
                        if self.apply_eyes_filter.get() and not attributes.get("eyes_open", False):
                            shutil.copyfile(src_path, os.path.join(rejected_folder, filename))
                            continue
                        if self.apply_smile_filter.get() and not attributes.get("smiling", False):
                            shutil.copyfile(src_path, os.path.join(rejected_folder, filename))
                            continue
                    except Exception as e:
                        self.output_box.insert(tk.END, f"⚠️ Face analysis failed on {filename}: {e}\n")
                        shutil.copyfile(src_path, os.path.join(rejected_folder, filename))
                        continue

                # Duplicate filter
                if self.apply_duplicate_filter.get():
                    img_hash = get_image_hash(img)
                    if any(are_images_duplicates(img_hash, h) for h in seen_hashes):
                        shutil.copyfile(src_path, os.path.join(rejected_folder, filename))
                        continue
                    seen_hashes.append(img_hash)

                # Face clustering (safe)
                try:
                    embedding = get_face_embedding(img)
                    if embedding is not None:
                        if any(np.linalg.norm(embedding - e) < 0.6 for e in known_embeddings):
                            shutil.copyfile(src_path, os.path.join(rejected_folder, filename))
                            continue
                        known_embeddings.append(embedding)
                except Exception as e:
                    self.output_box.insert(tk.END, f"⚠️ Embedding failed for {filename}: {e}\n")
                    shutil.copyfile(src_path, os.path.join(rejected_folder, filename))
                    continue

                # Export and show
                shutil.copyfile(src_path, os.path.join(approved_folder, filename))
                exported += 1
                self.output_box.insert(tk.END, f"✅ {filename}\n")

                pil_img = Image.fromarray(img[:, :, ::-1])
                pil_img.thumbnail((100, 100))
                thumb = ImageTk.PhotoImage(pil_img)
                self.thumbnails.append(thumb)
                img_id = self.thumbnail_canvas.create_image(x, y, anchor=tk.NW, image=thumb)
                self.thumb_map[img_id] = (img, filename)
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
        self.thumbnail_canvas.bind("<Button-1>", self.on_thumbnail_click)

    def on_thumbnail_click(self, event):
        x, y = event.x, event.y
        for img_id in self.thumb_map:
            coords = self.thumbnail_canvas.coords(img_id)
            if coords and coords[0] <= x <= coords[0] + 100 and coords[1] <= y <= coords[1] + 100:
                img, filename = self.thumb_map[img_id]
                self.show_full_image(img, filename)
                break

if __name__ == "__main__":
    root = tk.Tk()
    app = AilbumsApp(root)
    root.mainloop()


