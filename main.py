import os
import shutil
from core.sorter import sort_images_by_blur
from utils.image_loader import load_images_from_folder
from core.face_filter import detect_face_attributes

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

    print("\nAnalyzing faces...")
    exported = 0
    for filename, _ in sorted_results:
        if exported >= export_count:
            break

        src_path = os.path.join(folder, filename)
        img = images[filename]

        try:
            attributes = detect_face_attributes(img)
            if attributes["eyes_open"] and attributes["smiling"]:
                dst_path = os.path.join(approved_folder, filename)
                exported += 1
            else:
                dst_path = os.path.join(rejected_folder, filename)
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")
            dst_path = os.path.join(rejected_folder, filename)

        shutil.copyfile(src_path, dst_path)

    print(f"\n✅ Exported {exported} smiling, eyes-open photos to: {approved_folder}")
    print(f"❌ Rejected others to: {rejected_folder}")

if __name__ == "__main__":
    main()
