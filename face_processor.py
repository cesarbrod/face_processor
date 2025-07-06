#!/usr/bin/env python3
"""
Face-Centered Image Processor
Processes images to center faces and create 512x512 square crops
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import argparse

class FaceImageProcessor:
    def __init__(self):
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    def get_image_files(self, folder_path):
        """Get all image files from the input folder"""
        image_files = []
        folder = Path(folder_path)
        
        for file_path in folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                image_files.append(file_path)
        
        return image_files
    
    def detect_face_and_eyes(self, image):
        """Detect faces and eyes in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None, None
        
        # Use the largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        
        # Try to detect eyes within the face region
        face_roi = gray[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(face_roi)
        
        # Convert eye coordinates to full image coordinates
        eyes_full = [(x + ex, y + ey, ew, eh) for ex, ey, ew, eh in eyes]
        
        return face, eyes_full
    
    def calculate_crop_region(self, image, face, eyes):
        """Calculate 512x512 crop region that centers face and preserves most original content"""
        h, w = image.shape[:2]
        fx, fy, fw, fh = face
        
        # Determine face position (prefer eyes if available)
        if len(eyes) >= 2:
            # Sort eyes by x-coordinate 
            eyes_sorted = sorted(eyes, key=lambda e: e[0])
            left_eye = eyes_sorted[0]
            right_eye = eyes_sorted[-1]
            
            # Calculate eye center
            face_x = (left_eye[0] + left_eye[2]//2 + right_eye[0] + right_eye[2]//2) // 2
            face_y = (left_eye[1] + left_eye[3]//2 + right_eye[1] + right_eye[3]//2) // 2
        else:
            # Fallback to face center
            face_x = fx + fw // 2
            face_y = fy + fh // 2
        
        # Target position in 512x512 output:
        # - Face horizontally centered: x = 256
        # - Eyes in top third: y = 170 (512/3 ≈ 170)
        target_x = 256
        target_y = 170
        
        # Calculate where to center the 512x512 crop
        # Work backwards: if face should be at (256, 170) in final image,
        # and face is currently at (face_x, face_y), 
        # then crop should start at (face_x - 256, face_y - 170)
        crop_left = face_x - target_x
        crop_top = face_y - target_y
        crop_right = crop_left + 512
        crop_bottom = crop_top + 512
        
        # Adjust if crop goes outside image boundaries
        # Priority: keep the 512x512 size, shift the entire crop region
        if crop_left < 0:
            shift = -crop_left
            crop_left = 0
            crop_right = 512
        elif crop_right > w:
            shift = crop_right - w
            crop_right = w
            crop_left = w - 512
        
        if crop_top < 0:
            shift = -crop_top
            crop_top = 0
            crop_bottom = 512
        elif crop_bottom > h:
            shift = crop_bottom - h
            crop_bottom = h
            crop_top = h - 512
        
        # Final bounds check
        crop_left = max(0, crop_left)
        crop_top = max(0, crop_top)
        crop_right = min(w, crop_left + 512)
        crop_bottom = min(h, crop_top + 512)
        
        return int(crop_left), int(crop_top), int(crop_right), int(crop_bottom)
    
    def process_image(self, image_path, output_folder):
        """Process a single image"""
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not read {image_path}")
                return False
            
            # Detect face and eyes
            face, eyes = self.detect_face_and_eyes(image)
            
            if face is None:
                print(f"Warning: No face detected in {image_path}")
                return False
            
            # Calculate 512x512 crop region
            left, top, right, bottom = self.calculate_crop_region(image, face, eyes)
            
            # Crop the image to 512x512 (or as close as possible)
            cropped = image[top:bottom, left:right]
            
            # If crop is not exactly 512x512 due to image boundaries, resize it
            if cropped.shape[0] != 512 or cropped.shape[1] != 512:
                final_image = cv2.resize(cropped, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            else:
                final_image = cropped
            
            # Save the processed image
            output_path = Path(output_folder) / f"processed_{image_path.name}"
            cv2.imwrite(str(output_path), final_image)
            
            print(f"✓ Processed: {image_path.name}")
            return True
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return False
    
    def process_folder(self, input_folder, output_folder):
        """Process all images in the input folder"""
        # Get all image files
        image_files = self.get_image_files(input_folder)
        
        if not image_files:
            print("❌ Error: No images found in the input folder!")
            print(f"Supported formats: {', '.join(self.supported_formats)}")
            return False
        
        print(f"📁 Found {len(image_files)} images in input folder")
        
        # Create output folder if it doesn't exist
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output folder ready: {output_folder}")
        
        # Process each image
        successful = 0
        failed = 0
        
        for image_file in image_files:
            if self.process_image(image_file, output_folder):
                successful += 1
            else:
                failed += 1
        
        print(f"\n🎉 Processing complete!")
        print(f"✅ Successfully processed: {successful} images")
        if failed > 0:
            print(f"❌ Failed to process: {failed} images")
        
        return successful > 0

def main():
    print("🎯 Face-Centered Image Processor")
    print("=" * 40)
    
    # Get input and output folders from user
    input_folder = input("📂 Enter input folder path: ").strip()
    output_folder = input("📁 Enter output folder path: ").strip()
    
    # Validate input folder
    if not os.path.exists(input_folder):
        print(f"❌ Error: Input folder '{input_folder}' does not exist!")
        sys.exit(1)
    
    if not os.path.isdir(input_folder):
        print(f"❌ Error: '{input_folder}' is not a directory!")
        sys.exit(1)
    
    # Initialize processor
    processor = FaceImageProcessor()
    
    # Process the folder
    print("\n🔄 Starting processing...")
    success = processor.process_folder(input_folder, output_folder)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
