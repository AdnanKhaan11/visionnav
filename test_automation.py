import sys
import asyncio

sys.path.insert(0, "src")

from visionnav.platforms.desktop import DesktopPlatform


async def test():
    p = DesktopPlatform()
    w, h = p.get_screen_size()
    print(f"Screen size: {w}x{h}")

    print("\nOpening Notepad in 3 seconds...")
    print("Watch your screen!")
    import time

    time.sleep(3)

    # Press Win+R to open Run dialog
    print("Step 1: Opening Run dialog...")
    await p.execute_key("win+r")
    time.sleep(1)

    # Type notepad
    print("Step 2: Typing notepad...")
    await p.execute_type("notepad")
    time.sleep(0.5)

    # Press Enter
    print("Step 3: Pressing Enter...")
    await p.execute_key("enter")
    time.sleep(3)

    # Capture screen after
    print("Step 4: Capturing screen...")
    arr, meta = await p.capture()

    from visionnav.perception.ocr import OCREngine

    ocr = OCREngine()
    regions = ocr.run(arr)

    texts = [r.text for r in regions]
    print(f"\nText on screen after: {texts[:10]}")

    notepad_signs = ["Notepad", "notepad", "Format", "View"]
    if any(sign in texts for sign in notepad_signs):
        print("\nNotepad opened successfully!")
    else:
        print("\nNotepad not detected - check your screen")


asyncio.run(test())
