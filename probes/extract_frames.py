import imageio.v2 as imageio
from PIL import Image
r = imageio.get_reader("/opt/LIBERO/out/final6/episode.mp4")
n = r.count_frames()
for i in (int(n*0.1), int(n*0.5), int(n*0.92)):
    Image.fromarray(r.get_data(i)).save(f"/opt/LIBERO/final6_f{i}.png")
print("frames:", n)
