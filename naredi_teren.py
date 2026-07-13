from pathlib import Path
import numpy as np
from PIL import Image

# velikost slike heightfielda
nrow, ncol = 100, 100

# mreža koordinat
x = np.linspace(0, 4 * np.pi, ncol)
y = np.linspace(0, 4 * np.pi, nrow)
X, Y = np.meshgrid(x, y)

# preprost valovit teren
Z = 0.02 * np.sin(X) + 0.015 * np.cos(Y)
# normalizacija na [0, 1]
Z = (Z - Z.min()) / (Z.max() - Z.min())

# pretvorba v 16-bit
Z16 = (Z * 65535).astype(np.uint16)

# shrani
out_dir = Path("terrains")
out_dir.mkdir(exist_ok=True)
Image.fromarray(Z16).save(out_dir / "terrain16.png")

print("Shranjeno v terrains/terrain16.png")

