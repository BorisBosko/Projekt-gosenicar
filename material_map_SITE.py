from pathlib import Path
import numpy as np
from PIL import Image

# Logična materialna mapa za terrain_params() v G4 kodi.
# Barve morajo ostati točno take, ker jih Python koda prepozna po RGB pragovih:
#   rdeča  -> blato/zemlja:  mu=0.8, lambda=0.5
#   zelena -> asfalt:        mu=1.2, lambda=0.2
#   modra  -> pesek:         mu=0.5, lambda=0.9
width, height = 300, 300
img = np.zeros((height, width, 3), dtype=np.uint8)

BLATO = (255, 0, 0)
ASFALT = (0, 255, 0)
PESEK = (0, 0, 255)

# Isti vrstni red kot vizualna mapa: levo blato, sredina asfalt, desno pesek.
for y in range(height):
    for x in range(width):
        if x < width // 3:
            img[y, x] = BLATO
        elif x < 2 * width // 3:
            img[y, x] = ASFALT
        else:
            img[y, x] = PESEK

out_dir = Path("terrains")
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "material_map.png"
Image.fromarray(img).save(out_path)
print(f"Shranjeno: {out_path}")
