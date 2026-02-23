import os
import re
import sys
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

#!/usr/bin/env python3
"""
fast-labeling.py

Simple GUI for live webcam preview and quick labeling.
- Add labels which appear as buttons.
- Clicking a label button saves the current webcam frame into dataset/<label>/
- Filenames are automatically enumerated (0001.jpg, 0002.jpg, ...)

Dependencies:
- opencv-python
- pillow
"""


DATASET_DIR = "dataset"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FILENAME_TEMPLATE = "{:04d}.jpg"
NUM_PATTERN = re.compile(r"(\d+)\.\w+$")


class FastLabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fast Labeling")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Video capture
        self.current_camera_index = 0
        self.available_cameras = self._detect_cameras()
        
        self.cap = cv2.VideoCapture(self.current_camera_index)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", f"Could not open webcam (device {self.current_camera_index}).")
            root.destroy()
            return
        # try to set reasonable size
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        self.current_frame_bgr = None
        self.photo_image = None
        self.labels = {}  # label -> button

        self._build_ui()
        os.makedirs(DATASET_DIR, exist_ok=True)
        self._update_frame()

    def _detect_cameras(self):
        """Detect available camera indices."""
        cameras = []
        for i in range(10):  # Check first 10 indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cameras.append(i)
                cap.release()
        return cameras if cameras else [0]

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Left: video
        video_frame = ttk.Frame(main)
        video_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0)

        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(expand=True)

        # Right: controls
        control_frame = ttk.Frame(main, width=200)
        control_frame.grid(row=0, column=1, sticky="ns", padx=(8, 0))

        # Camera selection
        if len(self.available_cameras) > 1:
            camera_frame = ttk.Frame(control_frame)
            camera_frame.pack(fill=tk.X, pady=(0, 8))
            ttk.Label(camera_frame, text="Camera:").pack(side=tk.LEFT)
            self.camera_var = tk.StringVar(value=f"Device {self.current_camera_index}")
            camera_menu = ttk.Combobox(
                camera_frame, 
                textvariable=self.camera_var,
                values=[f"Device {i}" for i in self.available_cameras],
                state="readonly",
                width=10
            )
            camera_menu.pack(side=tk.LEFT, padx=(4, 0))
            camera_menu.bind("<<ComboboxSelected>>", self.switch_camera)
            
            sep_cam = ttk.Separator(control_frame, orient=tk.HORIZONTAL)
            sep_cam.pack(fill=tk.X, pady=4)

        add_label_frame = ttk.Frame(control_frame)
        add_label_frame.pack(fill=tk.X, pady=(0, 8))
        self.label_entry = ttk.Entry(add_label_frame)
        self.label_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        add_btn = ttk.Button(add_label_frame, text="Add Label", command=self.add_label)
        add_btn.pack(side=tk.LEFT, padx=(4, 0))

        sep = ttk.Separator(control_frame, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, pady=4)

        lbl = ttk.Label(control_frame, text="Labels (click to save):")
        lbl.pack(anchor="w")

        # scrollable area for label buttons
        buttons_canvas = tk.Canvas(control_frame, borderwidth=0, height=320)
        buttons_scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=buttons_canvas.yview)
        self.buttons_container = ttk.Frame(buttons_canvas)

        self.buttons_container.bind(
            "<Configure>",
            lambda e: buttons_canvas.configure(scrollregion=buttons_canvas.bbox("all"))
        )

        buttons_canvas.create_window((0, 0), window=self.buttons_container, anchor="nw")
        buttons_canvas.configure(yscrollcommand=buttons_scrollbar.set)

        buttons_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        buttons_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Footer: instructions and exit
        foot = ttk.Frame(control_frame)
        foot.pack(fill=tk.X, pady=(8, 0))
        info = ttk.Label(foot, text="Saved images in ./dataset/<label>/")
        info.pack(anchor="w")
        quit_btn = ttk.Button(foot, text="Quit", command=self.on_close)
        quit_btn.pack(anchor="e", pady=(4, 0))

    def switch_camera(self, event=None):
        """Switch to a different camera device."""
        selected = self.camera_var.get()
        # Extract device number from "Device X"
        try:
            new_index = int(selected.split()[-1])
            if new_index == self.current_camera_index:
                return
            
            # Release current camera
            if self.cap and self.cap.isOpened():
                self.cap.release()
            
            # Open new camera
            self.cap = cv2.VideoCapture(new_index)
            if not self.cap.isOpened():
                messagebox.showerror("Camera error", f"Could not open camera device {new_index}.")
                # Try to reopen previous camera
                self.cap = cv2.VideoCapture(self.current_camera_index)
                self.camera_var.set(f"Device {self.current_camera_index}")
                return
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.current_camera_index = new_index
            
        except (ValueError, IndexError):
            pass

    def add_label(self):
        text = self.label_entry.get().strip()
        if not text:
            return
        if text in self.labels:
            # already exists, focus its button (no duplicate)
            self.labels[text].focus_set()
            self.label_entry.delete(0, tk.END)
            return
        folder = os.path.join(DATASET_DIR, text)
        os.makedirs(folder, exist_ok=True)
        btn = ttk.Button(self.buttons_container, text=text, width=20, command=lambda t=text: self.save_image(t))
        btn.pack(pady=2, anchor="w")
        self.labels[text] = btn
        self.label_entry.delete(0, tk.END)

    def save_image(self, label):
        if self.current_frame_bgr is None:
            return
        folder = os.path.join(DATASET_DIR, label)
        os.makedirs(folder, exist_ok=True)

        # find next index by scanning existing numeric filenames
        existing = os.listdir(folder)
        max_idx = 0
        for fn in existing:
            m = NUM_PATTERN.search(fn)
            if m:
                try:
                    val = int(m.group(1))
                    if val > max_idx:
                        max_idx = val
                except ValueError:
                    pass
        next_idx = max_idx + 1
        filename = FILENAME_TEMPLATE.format(next_idx)
        path = os.path.join(folder, filename)
        # save BGR frame as JPEG
        cv2.imwrite(path, self.current_frame_bgr)
        # small visual feedback: briefly disable button text change
        btn = self.labels.get(label)
        if btn:
            old = btn.cget("text")
            btn.config(text=f"{old} ✓")
            self.root.after(300, lambda b=btn, t=old: b.config(text=t))

    def _update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_bgr = frame.copy()
            # convert for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame_rgb)
            # fit into widget width if needed
            w, h = pil.size
            max_w = FRAME_WIDTH
            max_h = FRAME_HEIGHT
            if w > max_w or h > max_h:
                pil = pil.resize((max_w, max_h), Image.ANTIALIAS)
            self.photo_image = ImageTk.PhotoImage(image=pil)
            self.video_label.configure(image=self.photo_image)
        # schedule next frame
        self.root.after(30, self._update_frame)

    def on_close(self):
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        self.root.quit()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = FastLabelingApp(root)
    # if camera failed, app likely destroyed already
    if not hasattr(app, "cap") or not app.cap.isOpened():
        return
    root.mainloop()


if __name__ == "__main__":
    main()