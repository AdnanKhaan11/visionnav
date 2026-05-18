"""Android automation via ADB (Android Debug Bridge)."""

from __future__ import annotations
import io
import numpy as np
import structlog
import adbutils
from PIL import Image
from visionnav.platforms.base import PlatformAdapter

log = structlog.get_logger(__name__)


class AndroidPlatform(PlatformAdapter):
    def __init__(self, serial: str | None = None) -> None:

        self._device = adbutils.adb.device(serial=serial)
        log.info("android_connected", serial=self._device.serial)

    async def capture(self) -> tuple[np.ndarray, dict]:

        img = Image.open(io.BytesIO(self._device.screencap())).convert("RGB")
        arr = np.array(img)
        meta = {"width": img.width, "height": img.height, "platform": "android"}
        return arr, meta

    async def get_ui_tree(self) -> list[dict]:
        return []  # TODO Phase 5: UIAutomator2 dump

    def get_screen_size(self) -> tuple[int, int]:
        info = self._device.window_size()
        return info.width, info.height

    async def execute_click(self, x: int, y: int, button: str = "left") -> bool:
        self._device.shell(f"input tap {x} {y}")
        return True

    async def execute_type(self, text: str) -> bool:
        escaped = text.replace(" ", "%s").replace("'", "\\'")
        self._device.shell(f"input text '{escaped}'")
        return True

    async def execute_scroll(self, x: int, y: int, direction: str, amount: int) -> bool:
        dist = amount * 300
        if direction == "down":
            self._device.shell(f"input swipe {x} {y} {x} {y-dist} 300")
        else:
            self._device.shell(f"input swipe {x} {y} {x} {y+dist} 300")
        return True

    async def execute_key(self, key_combo: str) -> bool:
        kmap = {"enter": "66", "back": "4", "home": "3", "tab": "61"}
        kc = kmap.get(key_combo.lower(), "")
        if kc:
            self._device.shell(f"input keyevent {kc}")
        return bool(kc)


# what is ADB ?
#
# ADB stands for Android Debug Bridge. It is a versatile
# command-line tool that allows you to communicate with an Android
# device (either an emulator or a physical device) from your computer.
#  ADB provides various functionalities, such as installing and debugging apps,
#  accessing the device's file system, and executing shell commands on the device.
# It is commonly used by developers for testing and debugging Android applications,
#  as well as by power users for advanced device management tasks.

#  it let python send command like .
# tap screen
# type text
# swipe
# press back
# take screenshot


# IMPORTANT ARCHITECTURE IDEA

# Notice something VERY important:
# Both:
# DesktopPlatform
# AndroidPlatform

# share SAME methods:
# capture()
# execute_click()
# execute_type()

# Why?
# Because AI should use SAME interface everywhere.

# This is called:
# POLYMORPHISM

# what is import io ?
# Used for:
# BYTE STREAM HANDLING
# WHAT ARE BYTES?

# Android screenshot comes as:

# Raw binary data

# NOT normal image object.

# Like:
# 101010101010001010...

# io.BytesIO
# Converts raw bytes into file-like object.

# REAL LIFE ANALOGY

# Imagine screenshot arrives as:

# Compressed package compress package  means the raw binary data

# BytesIO unwraps it so PIL can read it.


# NUMPY
# import numpy as np
# Used for image arrays.
# AI models usually want:
# NumPy arrays
# not PIL images.


# improvement list in this module:

# Missing / Future Improvements for AndroidPlatform
# Error Handling
# Handle ADB connection failures
# Handle device disconnects
# Handle unauthorized device state
# Handle shell command failures
# Handle screenshot capture errors
# Add retry logic for unstable devices
# Async Improvements
# Use run_in_executor() for blocking ADB operations
# Prevent screenshot capture from blocking event loop
# Prevent shell commands from freezing async system
# Logging Improvements
# Log taps/clicks
# Log swipe actions
# Log typed text safely
# Log key events
# Log execution failures
# Add performance timing logs
# UI Tree Support
# Implement UIAutomator2
# Parse Android accessibility tree
# Detect buttons/textboxes
# Extract clickable elements
# Support semantic UI understanding

# (Currently returns empty list.)

# Security / Safety
# Validate coordinates before tap
# Prevent out-of-screen actions
# Sanitize shell command inputs better
# Add automation safety limits
# Gesture Improvements
# Long press support
# Multi-touch gestures
# Pinch zoom
# Drag and drop
# Smooth scrolling
# Gesture duration customization
# Keyboard Improvements
# Expand keycode map
# Support key combinations
# Support volume/power/media keys
# Better Unicode typing support
# Better emoji support
# Device Management
# Multiple device support
# Auto device discovery
# Device reconnect logic
# Emulator detection
# Device capability detection
# Performance Improvements
# Faster screenshot pipeline
# Screenshot compression options
# Streaming screenshots
# Partial screen capture
# Frame caching
# Metadata Improvements
# Add orientation info
# Add DPI info
# Add device model
# Add Android version
# Add rotation state
# Architecture Improvements
# Centralized shell command helper
# Shared base utilities for platforms
# Action queue system
# Command timeout system
# Better abstraction for gestures
