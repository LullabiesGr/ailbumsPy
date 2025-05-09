import cv2

def get_blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def sort_images_by_blur(images):
    results = {}
    for filename, img in images.items():
        score = get_blur_score(img)
        results[filename] = score
    return results

