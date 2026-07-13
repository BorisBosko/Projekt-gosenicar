from pathlib import Path
import numpy as np
from PIL import Image

width, height = 300, 300
img = np.zeros((height, width, 3), dtype=np.uint8)

# vizualne barve
ASPHALT = (120, 120, 120)   # siva
BLATO   = (120, 70, 30)     # rjava
PESEK   = (220, 200, 90)    # rumena

for y in range(height):
    for x in range(width):
        if x < width // 3:
            img[y, x] = BLATO
        elif x < 2 * width // 3:
            img[y, x] = ASPHALT
        else:
            img[y, x] = PESEK

out_dir = Path("terrains")
out_dir.mkdir(exist_ok=True)

Image.fromarray(img).save(out_dir / "material_map_vis.png")
print("Shranjeno: terrains/material_map_vis.png")