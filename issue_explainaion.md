**31 passed. 0 failed. 🎉**

You just fixed your first real bug in this project. Remember this moment.

---

## Now Let Me Explain Your Question

You asked — where does `240 / 720` come from?

Open `D:\visionnav\tests\unit\test_coord_utils.py`

You will see this test:

```python
def test_round_trip():
    orig_x, orig_y = 320, 240
    nx, ny = normalize(orig_x, orig_y, 1280, 720)
    x, y = denormalize(nx, ny, 1280, 720)
    assert x == orig_x
    assert y == orig_y
```

Here `1280 x 720` is a screen resolution — standard HD screen size.

```
1280 = screen width  in pixels
720  = screen height in pixels
```

The test says:

```
I have a point at pixel (320, 240) on a 1280x720 screen
Normalize it  → convert to (0.25, 0.3333)  
Denormalize it → convert back to pixels
Result should be exactly (320, 240) again
```

This is called a **round trip test** — go forward and come back, you must land exactly where you started.

---

## Why Do We Normalize At All?

This is important. Understand this deeply.

```
Screen A = 1280 x 720   (your laptop)
Screen B = 1920 x 1080  (big monitor)
Screen C = 1080 x 1920  (phone)
```

If the model says **"click at pixel 640, 360"** — that means different things on each screen.

But if the model says **"click at 0.5, 0.5"** — that always means **the center of any screen**, no matter the size.

```
0.5 * 1280 = 640   ✅ center of laptop
0.5 * 1920 = 960   ✅ center of big monitor
0.5 * 1080 = 540   ✅ center of phone
```

**One sentence to remember:**
> We normalize coordinates so the model works on any screen size without retraining.

---

## Now Run The API Server

```bash
python -m uvicorn visionnav.api.app:create_app --factory --port 8000 --reload
```

Then open your browser and go to:

```
http://localhost:8000/docs
```

Tell me what you see.