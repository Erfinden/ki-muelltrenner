import os
import sys
import time
import threading
import cv2
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from pathlib import Path

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
ENTRY_WIDTH = 28
BUTTON_WIDTH = 12

# ── Arduino controller path ───────────────────────────────────────────────
_CONTROLLER_DIR = Path(__file__).parent.parent / "arduino-sketch"
if str(_CONTROLLER_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTROLLER_DIR))


class ToolTip:
    def __init__(self, widget, text_var):
        self.widget = widget
        self.text_var = text_var
        self.tip = None
        self.widget.bind("<Enter>", self._show)
        self.widget.bind("<Leave>", self._hide)

    def _show(self, event=None):
        if self.tip or not self.text_var.get():
            return
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.attributes("-topmost", True)
        label = ttk.Label(self.tip, text=self.text_var.get(), padding=(6, 3))
        label.pack()
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip.wm_geometry(f"+{x}+{y}")

    def _hide(self, event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class FastLabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fast Labeling")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.icon_image = None
        self._set_window_icon()

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
        self.last_saved_path = None
        self.last_saved_label = None
        self.remove_last_btn = None
        self.status_var = tk.StringVar(value="")
        self.dataset_dir = tk.StringVar(value=self._default_dataset_dir())
        self.buttons_canvas = None
        self.buttons_scrollbar = None
        self.scrollbar_visible = False

        # ── Arduino / LED state ────────────────────────────────────────────
        self._arduino      = None
        self._arduino_lock = threading.Lock()
        self._led_on       = False

        self._build_ui()
        os.makedirs(self.dataset_dir.get(), exist_ok=True)
        self._load_existing_labels()
        self._bring_to_front()
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

        dataset_frame = ttk.Frame(control_frame)
        dataset_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(dataset_frame, text="Dataset:").pack(anchor="w")
        dataset_entry = ttk.Entry(
            dataset_frame,
            textvariable=self.dataset_dir,
            width=ENTRY_WIDTH,
            state="readonly",
        )
        dataset_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ToolTip(dataset_entry, self.dataset_dir)
        browse_btn = ttk.Button(
            dataset_frame,
            text="Browse",
            width=BUTTON_WIDTH,
            command=self.browse_dataset_dir,
        )
        browse_btn.pack(side=tk.LEFT, padx=(4, 0))

        add_label_frame = ttk.Frame(control_frame)
        add_label_frame.pack(fill=tk.X, pady=(0, 8))
        self.label_entry = ttk.Entry(add_label_frame, width=ENTRY_WIDTH)
        self.label_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        add_btn = ttk.Button(
            add_label_frame,
            text="Add Label",
            width=BUTTON_WIDTH,
            command=self.add_label,
        )
        add_btn.pack(side=tk.LEFT, padx=(4, 0))

        remove_frame = ttk.Frame(control_frame)
        remove_frame.pack(fill=tk.X, pady=(0, 8))
        self.remove_last_btn = ttk.Button(
            remove_frame,
            text="Remove Last Image",
            command=self.remove_last_image,
            state=tk.DISABLED,
        )
        self.remove_last_btn.pack(anchor="w")
        status_lbl = ttk.Label(remove_frame, textvariable=self.status_var)
        status_lbl.pack(anchor="w", pady=(4, 0))

        # ── LED / Arduino panel ───────────────────────────────────────────
        led_frame = ttk.LabelFrame(control_frame, text="Beleuchtung")
        led_frame.pack(fill=tk.X, pady=(0, 8))

        arduino_row = ttk.Frame(led_frame)
        arduino_row.pack(fill=tk.X, padx=4, pady=(4, 0))
        self._ard_status_lbl = ttk.Label(arduino_row, text="● Nicht verbunden",
                                         foreground="#f87171")
        self._ard_status_lbl.pack(side=tk.LEFT)

        self._connect_btn = ttk.Button(
            led_frame, text="Verbinden", command=self._toggle_arduino_connection
        )
        self._connect_btn.pack(fill=tk.X, padx=4, pady=(4, 2))

        self._light_btn = ttk.Button(
            led_frame,
            text="💡  Licht: AUS",
            state=tk.DISABLED,
            command=self._toggle_led,
        )
        self._light_btn.pack(fill=tk.X, padx=4, pady=(2, 6))

        sep = ttk.Separator(control_frame, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, pady=4)

        labels_header = ttk.Frame(control_frame)
        labels_header.pack(fill=tk.X)
        lbl = ttk.Label(labels_header, text="Labels (click to save):")
        lbl.pack(anchor="w")

        # scrollable area for label buttons
        list_frame = ttk.Frame(control_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        list_frame.columnconfigure(0, weight=1)

        self.buttons_canvas = tk.Canvas(list_frame, borderwidth=0, highlightthickness=0, height=320)
        self.buttons_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.buttons_canvas.yview)
        self.buttons_container = ttk.Frame(self.buttons_canvas)

        self.buttons_container.bind("<Configure>", self._on_buttons_configure)
        self.buttons_canvas.bind("<Configure>", self._on_canvas_configure)

        self.buttons_canvas.create_window((0, 0), window=self.buttons_container, anchor="nw")
        self.buttons_canvas.configure(yscrollcommand=self.buttons_scrollbar.set)

        self.buttons_canvas.grid(row=0, column=0, sticky="nsew")
        self._update_scrollbar_visibility()
        self._bind_mousewheel(self.buttons_canvas)

        # Footer removed for cleaner layout

    def _set_window_icon(self):
        icon_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "fast_labeling_icon.png")
        )
        if not os.path.exists(icon_path):
            return
        try:
            icon_image = ImageTk.PhotoImage(Image.open(icon_path))
            self.root.iconphoto(True, icon_image)
            self.icon_image = icon_image
        except Exception:
            pass

    def _bring_to_front(self):
        self.root.update_idletasks()
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(200, lambda: self.root.attributes("-topmost", False))
        self.root.focus_force()

    def _on_buttons_configure(self, event=None):
        if self.buttons_canvas:
            self.buttons_canvas.configure(scrollregion=self.buttons_canvas.bbox("all"))
            self._update_scrollbar_visibility()

    def _on_canvas_configure(self, event=None):
        self._update_scrollbar_visibility()

    def _update_scrollbar_visibility(self):
        if not self.buttons_canvas or not self.buttons_scrollbar:
            return
        bbox = self.buttons_canvas.bbox("all")
        if not bbox:
            return
        content_height = bbox[3] - bbox[1]
        canvas_height = self.buttons_canvas.winfo_height()
        needs_scrollbar = content_height > canvas_height
        if needs_scrollbar and not self.scrollbar_visible:
            self.buttons_scrollbar.grid(row=0, column=1, sticky="ns")
            self.scrollbar_visible = True
            self.buttons_canvas.configure(yscrollcommand=self.buttons_scrollbar.set)
        elif not needs_scrollbar and self.scrollbar_visible:
            self.buttons_scrollbar.grid_remove()
            self.scrollbar_visible = False
            self.buttons_canvas.configure(yscrollcommand=None)

    def _bind_mousewheel(self, widget):
        widget.bind_all("<MouseWheel>", self._on_mousewheel)
        widget.bind_all("<Button-4>", self._on_mousewheel)
        widget.bind_all("<Button-5>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        if not self.buttons_canvas or not self.scrollbar_visible:
            return
        if event.num == 4:
            delta = -120
        elif event.num == 5:
            delta = 120
        else:
            delta = -1 * event.delta
        self.buttons_canvas.yview_scroll(int(delta / 120), "units")

    def _default_dataset_dir(self):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), DATASET_DIR))

    def browse_dataset_dir(self):
        selected = filedialog.askdirectory(initialdir=self.dataset_dir.get())
        if selected:
            self._set_dataset_dir(selected)

    def _set_dataset_dir(self, path):
        new_dir = os.path.abspath(path)
        if new_dir == self.dataset_dir.get():
            return
        self.dataset_dir.set(new_dir)
        os.makedirs(new_dir, exist_ok=True)
        self._reset_last_saved()
        self._reload_labels()

    def _reset_last_saved(self):
        self.last_saved_path = None
        self.last_saved_label = None
        if self.remove_last_btn:
            self.remove_last_btn.config(state=tk.DISABLED)

    def switch_camera(self, event=None):
        """Switch to a different camera device."""
        selected = self.camera_var.get()
        # Extract device number from "Device X"
        try:
            new_index = int(selected.split()[-1])
            if new_index == self.current_camera_index:
                return
            # Open new camera and validate before swapping
            new_cap = cv2.VideoCapture(new_index)
            if not new_cap.isOpened():
                messagebox.showerror("Camera error", f"Could not open camera device {new_index}.")
                self.camera_var.set(f"Device {self.current_camera_index}")
                return

            new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            ret, _ = new_cap.read()
            if not ret:
                new_cap.release()
                messagebox.showerror("Camera error", f"Camera device {new_index} is not available.")
                self.camera_var.set(f"Device {self.current_camera_index}")
                return

            # Swap cameras
            if self.cap and self.cap.isOpened():
                self.cap.release()
            self.cap = new_cap
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
        self._create_label_button(text)
        self.label_entry.delete(0, tk.END)

    def _load_existing_labels(self):
        try:
            entries = os.listdir(self.dataset_dir.get())
        except FileNotFoundError:
            return
        for name in sorted(entries):
            folder = os.path.join(self.dataset_dir.get(), name)
            if os.path.isdir(folder) and name not in self.labels:
                self._create_label_button(name)

    def _reload_labels(self):
        for child in self.buttons_container.winfo_children():
            child.destroy()
        self.labels.clear()
        self._load_existing_labels()

    def _create_label_button(self, label):
        folder = os.path.join(self.dataset_dir.get(), label)
        os.makedirs(folder, exist_ok=True)
        btn = ttk.Button(self.buttons_container, text=label, width=20, command=lambda t=label: self.save_image(t))
        btn.pack(pady=2, anchor="w")
        self.labels[label] = btn

    def save_image(self, label):
        if self.current_frame_bgr is None:
            return
        folder = os.path.join(self.dataset_dir.get(), label)
        os.makedirs(folder, exist_ok=True)

        # Filename: <label>_YYYYMMDDHHMMSS[_µs].jpg  (matches predict_gui convention)
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{label}_{ts}.jpg"
        path = os.path.join(folder, filename)
        # Avoid collision if two saves happen within the same second
        if os.path.exists(path):
            µs = time.strftime("%f") if hasattr(time, "strftime") else str(int(time.time() * 1e6) % 1_000_000)
            filename = f"{label}_{ts}_{µs}.jpg"
            path = os.path.join(folder, filename)
        # save BGR frame as JPEG
        cv2.imwrite(path, self.current_frame_bgr)
        self.last_saved_path = path
        self.last_saved_label = label
        if self.remove_last_btn:
            self.remove_last_btn.config(state=tk.NORMAL)
        self.status_var.set(f"{label}/{filename} saved")
        # small visual feedback: briefly flash a checkmark on the button
        btn = self.labels.get(label)
        if btn:
            old = btn.cget("text")
            btn.config(text=f"{old} ✓")
            self.root.after(300, lambda b=btn, t=old: b.config(text=t))

    def remove_last_image(self):
        if not self.last_saved_path:
            return
        try:
            if os.path.exists(self.last_saved_path):
                os.remove(self.last_saved_path)
        finally:
            removed_name = os.path.basename(self.last_saved_path)
            removed_label = self.last_saved_label
            self.last_saved_path = None
            self.last_saved_label = None
            if self.remove_last_btn:
                self.remove_last_btn.config(state=tk.DISABLED)
            if removed_label:
                self.status_var.set(f"{removed_label}/{removed_name} removed")
            else:
                self.status_var.set(f"{removed_name} removed")

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
                try:
                    resample = Image.Resampling.LANCZOS
                except AttributeError:
                    resample = Image.LANCZOS
                pil = pil.resize((max_w, max_h), resample)
            self.photo_image = ImageTk.PhotoImage(image=pil)
            self.video_label.configure(image=self.photo_image)
        # schedule next frame
        self.root.after(30, self._update_frame)

    def on_close(self):
        # Turn off LED and disconnect gracefully
        with self._arduino_lock:
            if self._arduino is not None:
                try:
                    self._arduino.led_off()
                except Exception:
                    pass
                try:
                    self._arduino.close()
                except Exception:
                    pass
                self._arduino = None
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        self.root.quit()
        self.root.destroy()

    # ── Arduino / LED helpers ─────────────────────────────────────────

    def _toggle_arduino_connection(self):
        with self._arduino_lock:
            if self._arduino is not None:
                self._disconnect_arduino()
            else:
                self._connect_arduino()

    def _connect_arduino(self):
        """Try to connect to the Arduino (called with _arduino_lock held)."""
        try:
            from trash_bin_controller import TrashBinController
        except ImportError:
            messagebox.showerror(
                "Importfehler",
                "trash_bin_controller.py nicht gefunden.\n"
                f"Erwartet in: {_CONTROLLER_DIR}"
            )
            return

        self._ard_status_lbl.config(text="● Verbinde …", foreground="#facc15")
        self._connect_btn.config(state=tk.DISABLED)
        self.root.update_idletasks()

        try:
            ctrl = TrashBinController()
            self._arduino = ctrl
            self._ard_status_lbl.config(
                text=f"● Verbunden ({ctrl._serial.name})", foreground="#4ade80"
            )
            self._connect_btn.config(text="Trennen", state=tk.NORMAL)
            self._light_btn.config(state=tk.NORMAL)
        except Exception as exc:
            self._arduino = None
            self._ard_status_lbl.config(text="● Nicht verbunden", foreground="#f87171")
            self._connect_btn.config(state=tk.NORMAL)
            messagebox.showerror("Verbindungsfehler", f"Arduino nicht erreichbar:\n{exc}")

    def _disconnect_arduino(self):
        """Disconnect from the Arduino (called with _arduino_lock held)."""
        try:
            if self._arduino:
                if self._led_on:
                    self._arduino.led_off()
                self._arduino.close()
        except Exception:
            pass
        self._arduino = None
        self._led_on = False
        self._ard_status_lbl.config(text="● Nicht verbunden", foreground="#f87171")
        self._connect_btn.config(text="Verbinden", state=tk.NORMAL)
        self._light_btn.config(text="💡  Licht: AUS", state=tk.DISABLED)

    def _toggle_led(self):
        """Toggle the LED strip on/off."""
        with self._arduino_lock:
            if self._arduino is None:
                return
            try:
                if self._led_on:
                    self._arduino.led_off()
                    self._led_on = False
                else:
                    self._arduino.led_on()
                    self._led_on = True
            except Exception as exc:
                self._arduino = None
                self._led_on = False
                self.root.after(0, self._on_arduino_lost)
                messagebox.showerror("Arduino-Fehler", str(exc))
                return
        text = "💡  Licht: AN" if self._led_on else "💡  Licht: AUS"
        self._light_btn.config(text=text)

    def _on_arduino_lost(self):
        """Reset UI when Arduino connection is unexpectedly lost."""
        self._ard_status_lbl.config(text="● Verbindung verloren", foreground="#f87171")
        self._connect_btn.config(text="Verbinden", state=tk.NORMAL)
        self._light_btn.config(text="💡  Licht: AUS", state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = FastLabelingApp(root)
    # if camera failed, app likely destroyed already
    if not hasattr(app, "cap") or not app.cap.isOpened():
        return
    root.mainloop()


if __name__ == "__main__":
    main()