#!/usr/bin/env python3
"""
Check available camera devices
"""
import cv2

print("Checking for available camera devices...")
print("-" * 50)

found_camera = False
for i in range(10):  # Check first 10 device indices
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✓ Camera found at device index {i}")
            print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
            found_camera = True
        cap.release()

if not found_camera:
    print("\n✗ No camera devices found.")
    print("\nPossible solutions:")
    print("1. Check if another application is using the camera")
    print("2. On macOS, grant Terminal camera permissions:")
    print("   System Preferences > Security & Privacy > Camera")
    print("3. Try running with a different Python executable")
    print("4. Check if the camera is properly connected")
else:
    print("\n" + "-" * 50)
    print("Camera check complete!")
