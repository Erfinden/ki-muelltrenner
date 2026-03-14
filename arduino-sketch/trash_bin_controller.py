"""
trash_bin_controller.py
=======================
Python interface for the Smart Trash Bin Arduino sketch.

Dependencies:  pip install pyserial

Usage example
-------------
    from trash_bin_controller import TrashBinController

    with TrashBinController(port="/dev/ttyUSB0") as bin_ctrl:
        bin_ctrl.open_lid(1)
        time.sleep(3)
        bin_ctrl.close_lid(1)
        bin_ctrl.led_on()
        time.sleep(2)
        bin_ctrl.led_off()
"""

import time
import serial
import threading
import queue
from serial.tools import list_ports


def find_arduino_port() -> str | None:
    """Auto-detect the first connected Arduino / CH340 / CP210x device."""
    for port in list_ports.comports():
        desc = (port.description or "").lower()
        if any(kw in desc for kw in ("arduino", "ch340", "cp210", "ftdi", "usb serial")):
            return port.device
    return None


class TrashBinController:
    """
    Controls the smart trash bin lids over USB serial.

    Parameters
    ----------
    port : str | None
        Serial port, e.g. '/dev/ttyUSB0' or 'COM3'.
        Pass None to auto-detect.
    baud : int
        Must match Serial.begin() in the Arduino sketch (default 9600).
    timeout : float
        Read timeout in seconds.
    """

    READY_TIMEOUT = 5.0

    def __init__(self, port: str | None = None, baud: int = 9600, timeout: float = 2.0):
        if port is None:
            port = find_arduino_port()
            if port is None:
                raise RuntimeError(
                    "Could not auto-detect an Arduino. "
                    "Pass the port explicitly, e.g. TrashBinController(port='/dev/ttyUSB0')"
                )
        self._serial = serial.Serial(port, baud, timeout=timeout)
        self._wait_for_ready()

        self._stop_event = threading.Event()
        self._response_queue = queue.Queue()
        self._callbacks = {}
        
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if hasattr(self, '_stop_event'):
            self._stop_event.set()
        if self._serial.is_open:
            self._serial.close()

    # ── Public API ─────────────────────────────────────────────────────────

    def register_callback(self, btn_name: str, callback):
        """Register a callback for button events from the Arduino."""
        self._callbacks[btn_name] = callback

    def open_lid(self, lid: int | str) -> str:
        """
        Open a lid.

        Parameters
        ----------
        lid : 1, 2, or 'ALL'
        """
        return self._send(f"OPEN {lid}")

    def close_lid(self, lid: int | str) -> str:
        """
        Close a lid.

        Parameters
        ----------
        lid : 1, 2, or 'ALL'
        """
        return self._send(f"CLOSE {lid}")

    def led_on(self) -> str:
        """Turn the LED strip on (sends 'LED ON')."""
        return self._send("LED ON")

    def led_off(self) -> str:
        """Turn the LED strip off (sends 'LED OFF')."""
        return self._send("LED OFF")

    # ── Internal ───────────────────────────────────────────────────────────

    def _reader_loop(self):
        while not self._stop_event.is_set():
            if self._serial.in_waiting > 0:
                try:
                    line = self._serial.readline().decode("ascii", errors="ignore").strip()
                    if not line:
                        continue
                    if line.startswith("BTN_"):
                        cb = self._callbacks.get(line)
                        if cb:
                            cb()
                    else:
                        self._response_queue.put(line)
                except Exception:
                    pass
            else:
                time.sleep(0.01)

    def _send(self, command: str) -> str:
        self._serial.write((command + "\n").encode("ascii"))
        try:
            response = self._response_queue.get(timeout=2.0)
            return response
        except queue.Empty:
            return "TIMEOUT"

    def _wait_for_ready(self):
        deadline = time.time() + self.READY_TIMEOUT
        while time.time() < deadline:
            line = self._serial.readline().decode("ascii", errors="ignore").strip()
            if line == "READY":
                return
        raise TimeoutError(
            f"Arduino did not send 'READY' within {self.READY_TIMEOUT}s "
            f"on {self._serial.name}"
        )


# ── Quick smoke test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    port_arg = sys.argv[1] if len(sys.argv) > 1 else None

    print("Connecting to Arduino…")
    with TrashBinController(port=port_arg) as ctrl:
        print("Opening lid 1…")
        print(ctrl.open_lid(1))
        time.sleep(2)

        print("Closing lid 1…")
        print(ctrl.close_lid(1))
        time.sleep(1)

        print("Opening lid 2…")
        print(ctrl.open_lid(2))
        time.sleep(2)

        print("Closing all lids…")
        print(ctrl.close_lid("ALL"))
