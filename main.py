import os
import shutil
from core.sorter import sort_images_by_blur
from utils.image_loader import load_images_from_folder

def main():
    folder = input("Enter path to image folder: ")
    if not os.path.isdir(folder):
        print("Invalid folder.")
        return

    print("Loading images...")
    images = load_images_from_folder(folder)

    print("Sorting images by sharpness...")
    results = sort_images_by_blur(images)

    # Sort by sharpness (higher Laplacian = sharper)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    print("\nTop sharpest photos:")
    for filename, score in sorted_results[:10]:
        print(f"{filename}: {score:.2f}")

    # Export top 50 to Approved folder
    export_count = 50
    approved_folder = os.path.join(folder, "Approved")
    os.makedirs(approved_folder, exist_ok=True)

    for filename, _ in sorted_results[:export_count]:
        src_path = os.path.join(folder, filename)
        dst_path = os.path.join(approved_folder, filename)
        shutil.copyfile(src_path, dst_path)

    print(f"\n✅ Exported top {export_count} sharpest photos to: {approved_folder}")

if __name__ == "__main__":
    main()

