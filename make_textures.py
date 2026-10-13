"""Generate procedural textures for the pretty scene (wood, tiles)."""
import numpy as np
from PIL import Image

rng = np.random.default_rng(7)

# ---- wood table top: warm brown with horizontal grain ----
W, H = 512, 512
x = np.linspace(0, 1, W)[None, :].repeat(H, 0)
y = np.linspace(0, 1, H)[:, None].repeat(W, 1)
grain = 0.5 + 0.5 * np.sin(y * 60 + 3 * np.sin(y * 13) + 2 * np.sin(x * 5))
grain = grain ** 0.7
noise = rng.normal(0, 0.06, (H, W))
fine = 0.5 + 0.5 * np.sin(y * 400 + rng.normal(0, 1, (H, W)) * 2)
v = 0.55 + 0.30 * grain + 0.08 * fine + noise
v = np.clip(v, 0, 1)
r = (v * 150 + 45).astype(np.uint8)
g = (v * 100 + 28).astype(np.uint8)
b = (v * 60 + 16).astype(np.uint8)
Image.fromarray(np.stack([r, g, b], -1)).save("tex_wood.png")

# ---- floor: large soft-gray tiles ----
T = 512
tiles = np.zeros((T, T, 3), dtype=np.uint8)
n = 4  # 4x4 tiles
for i in range(n):
    for j in range(n):
        base = 205 + ((i + j) % 2) * 12
        tiles[i * T // n:(i + 1) * T // n, j * T // n:(j + 1) * T // n] = base
noise = rng.normal(0, 5, (T, T, 1)).astype(np.int16)
tiles = np.clip(tiles.astype(np.int16) + noise, 0, 255).astype(np.uint8)
# grout lines
for k in range(n + 1):
    p = k * T // n
    tiles[max(p - 2, 0):p + 2, :] = 150
    tiles[:, max(p - 2, 0):p + 2] = 150
Image.fromarray(tiles).save("tex_tiles.png")

# ---- wall: very light warm gray with subtle vertical gradient ----
W2, H2 = 256, 256
gy = np.linspace(225, 190, H2)[:, None].repeat(W2, 1)
wall = np.stack([gy, gy - 3, gy - 8], -1).clip(0, 255).astype(np.uint8)
Image.fromarray(wall).save("tex_wall.png")

print("textures written")
