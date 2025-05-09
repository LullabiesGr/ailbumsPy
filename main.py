import os
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

    print("\nTop sharpest photos:")
    for filename, score in sorted(results.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{filename}: {score:.2f}")

if __name__ == "__main__":
    main()

