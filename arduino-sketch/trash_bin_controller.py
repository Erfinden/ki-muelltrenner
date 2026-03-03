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
"""

import time
import serial
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

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self._serial.is_open:
            self._serial.close()

    # ── Public API ─────────────────────────────────────────────────────────

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

    # ── Internal ───────────────────────────────────────────────────────────

    def _send(self, command: str) -> str:
        self._serial.write((command + "\n").encode("ascii"))
        response = self._serial.readline().decode("ascii", errors="ignore").strip()
        return response

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
