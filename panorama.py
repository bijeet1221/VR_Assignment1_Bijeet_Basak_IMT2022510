import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt

# Ensure output directory exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

image_path = glob.glob("images/*.jpeg")
images = []

for image in image_path:
    img = cv2.imread(image)
    images.append(img)
    cv2.imshow("Image", img)
    cv2.waitKey(1000)

sift = cv2.SIFT_create()
imageStitcher = cv2.Stitcher_create()
error, stitched_img = imageStitcher.stitch(images)

if not error:
    gray_stitched = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2GRAY)
    keypoints_stitched, descriptors_stitched = sift.detectAndCompute(gray_stitched, None)
    stitched_with_keypoints = cv2.drawKeypoints(stitched_img, keypoints_stitched, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Save stitched images in output directory
    stitched_path = os.path.join(output_dir, "stitched_image.jpeg")
    keypoints_path = os.path.join(output_dir, "stitched_with_keypoints.jpeg")
    
    cv2.imwrite(stitched_path, stitched_img)
    cv2.imwrite(keypoints_path, stitched_with_keypoints)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Stitched Image Without Keypoints
    stitched_rgb = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(stitched_rgb)
    axes[0].axis("off")
    axes[0].set_title("Stitched Image (Without Keypoints)")

    # Stitched Image With Keypoints
    stitched_with_keypoints_rgb = cv2.cvtColor(stitched_with_keypoints, cv2.COLOR_BGR2RGB)
    axes[1].imshow(stitched_with_keypoints_rgb)
    axes[1].axis("off")
    axes[1].set_title("Stitched Image (With Keypoints)")

    plt.show()
    
cv2.destroyAllWindows()
