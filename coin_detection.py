import cv2
import numpy as np
import os

def detect_coins(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # Edge Detection using Canny
    edges = cv2.Canny(blurred, 30, 150)
    
    # Find contours from the edges detected
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours around detected coins
    detected_coins = image.copy()
    cv2.drawContours(detected_coins, contours, -1, (0, 255, 0), 2)
    
    return detected_coins, contours, image

def segment_coins(image, contours):
    segmented_coins = []
    
    for i, contour in enumerate(contours):
        # Get bounding box around each coin
        x, y, w, h = cv2.boundingRect(contour)
        
        # Create a mask for the individual coin
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Extract the region of interest (ROI) using the bounding box
        coin_roi = image[y:y+h, x:x+w]
        mask_roi = mask[y:y+h, x:x+w]

        # Apply bitwise_and to extract only the coin from the bounding box
        segmented_coin = cv2.bitwise_and(coin_roi, coin_roi, mask=mask_roi)

        # Append the segmented coin
        segmented_coins.append(segmented_coin)

    return segmented_coins

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

image_path = "coins.jpeg"  
detected_coins, contours, image = detect_coins(image_path)
segmented_coins = segment_coins(image, contours)
coin_count = len(contours)

print(f"Total number of coins detected: {coin_count}")
cv2.imshow("Detected Coins", detected_coins) # Display the detected coins
cv2.imwrite(os.path.join(output_dir, "Detected_coins.jpeg"), detected_coins)

for i, coin in enumerate(segmented_coins):  # Display segmented coins
    cv2.imshow(f"Coin {i+1}", coin)
    
cv2.waitKey(0)
cv2.destroyAllWindows()