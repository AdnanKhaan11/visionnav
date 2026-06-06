from PIL import Image, ImageDraw

# Screenshot path
# image_path = r"D:\visionnav\data\recordings\screenshots\9074fe76\step_000.png"
image_path = r"D:\visionnav\data\recordings\screenshots\176c5b10\step_003.png"
# Normalized coordinates from recorder
# ,
x_norm = 0.7167
y_norm = 0.1926

# Load image
img = Image.open(image_path)
width, height = img.size

# Convert normalized coordinates to pixels
x = int(x_norm * width)
y = int(y_norm * height)

print(f"Image size: {width}x{height}")
print(f"Pixel position: ({x}, {y})")

draw = ImageDraw.Draw(img)

# Draw crosshair
draw.line([(x - 40, y), (x + 40, y)], fill="red", width=4)
draw.line([(x, y - 40), (x, y + 40)], fill="red", width=4)

# Draw circle
radius = 20
draw.ellipse(
    [(x - radius, y - radius), (x + radius, y + radius)], outline="red", width=4
)

# Label
draw.text((x + 25, y - 25), f"({x}, {y})", fill="red")

# Save output
output_path = r"D:\visionnav\data\external\step_000_marked.png"
img.save(output_path)

print(f"Saved: {output_path}")
