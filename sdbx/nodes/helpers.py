import re

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

def conditioning_set_values(conditioning, values={}):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            n[1][k] = values[k]
        c.append(n)

    return c

def pillow(fn, arg):
    prev_value = None
    try:
        x = fn(arg)
    except (OSError, UnidentifiedImageError, ValueError): #PIL issues #4472 and #2445, also fixes ComfyUI issue #3416
        prev_value = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        x = fn(arg)
    finally:
        if prev_value is not None:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev_value
        return x

def generate_gradient(width, height):
    # Create random colors for the gradient
    c1 = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
    c2 = (255 - c1[0], 255 - c1[1], 255 - c1[2])

    # Generate a gradient image
    img = Image.new("RGB", (width, height), "white")
    for x in range(width):
        for y in range(height):
            # Determine the color based on x and y coordinates
            r = int((x + y) / (width + height) * (c2[0] - c1[0]) + c1[0])
            g = int((x + y) / (width + height) * (c2[1] - c1[1]) + c1[1])
            b = int((x + y) / (width + height) * (c2[2] - c1[2]) + c1[2])
            pixel = (r, g, b)
            img.putpixel((x, y), pixel)

    return img

def rename_class(base, name):
    # Create a new class dynamically, inheriting from base_class
    new = type(name, (base,), {})
    
    # Set the __name__ and __qualname__ attributes to reflect the new name
    new.__name__ = name
    new.__qualname__ = name
    
    return new

def format_name(name):
    return ' '.join(word[0].upper() + word[1:] if word else '' for word in re.split(r'_', name))
