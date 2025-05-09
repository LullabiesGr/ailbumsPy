
import os
import shutil
import numpy as np
from core.sorter import sort_images_by_blur
from utils.image_loader import load_images_from_folder
from core.face_filter import detect_face_attributes
from core.face_cluster import get_face_embedding, get_image_hash, are_images_duplicates

def main():
    folder = input("Enter path to image folder: ")
    if not os.path.isdir(folder):
        print("Invalid folder.")
        return

    print("Loading images...")
    images = load_images_from_folder(folder)

    print("Sorting images by sharpness...")
    results = sort_images_by_blur(images)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    print("\nTop sharpest photos:")
    for filename, score in sorted_results[:10]:
        print(f"{filename}: {score:.2f}")

    # Create export folders
    export_count = 50
    approved_folder = os.path.join(folder, "Approved")
    rejected_folder = os.path.join(folder, "Rejected")
    os.makedirs(approved_folder, exist_ok=True)
    os.makedirs(rejected_folder, exist_ok=True)

    print("\nAnalyzing faces and filtering...")
    exported = 0
    seen_hashes = []
    known_embeddings = []

    for filename, _ in sorted_results:
        if exported >= export_count:
            break

        src_path = os.path.join(folder, filename)
        img = images[filename]

        try:
            # Filter: eyes open + smiling
            attributes = detect_face_attributes(img)
            if not (attributes["eyes_open"] and attributes["smiling"]):
                dst = os.path.join(rejected_folder, filename)
                shutil.copyfile(src_path, dst)
                continue

            # Filter: duplicates
            img_hash = get_image_hash(img)
            if any(are_images_duplicates(img_hash, h) for h in seen_hashes):
                dst = os.path.join(rejected_folder, filename)
                shutil.copyfile(src_path, dst)
                continue
            seen_hashes.append(img_hash)

            # Filter: already seen person (based on embedding)
            embedding = get_face_embedding(img)
            if embedding is not None:
                if any(np.linalg.norm(embedding - e) < 0.6 for e in known_embeddings):
                    dst = os.path.join(rejected_folder, filename)
                    shutil.copyfile(src_path, dst)
                    continue
                known_embeddings.append(embedding)

            # Passed all filters
            dst = os.path.join(approved_folder, filename)
            shutil.copyfile(src_path, dst)
            exported += 1

        except Exception as e:
            print(f"Error with {filename}: {e}")
            dst = os.path.join(rejected_folder, filename)
            shutil.copyfile(src_path, dst)

    print(f"\n✅ Exported {exported} unique, smiling, eyes-open, sharp photos to: {approved_folder}")
    print(f"❌ Other photos moved to: {rejected_folder}")

if __name__ == "__main__":
    main()
