"""Extract franka_emika_panda from menagerie.zip into project root."""
import zipfile, os, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
z = zipfile.ZipFile(os.path.join(ROOT, "menagerie.zip"))
names = [n for n in z.namelist() if "franka_emika_panda/" in n and not n.endswith("/")]
for n in names:
    rel = n.split("franka_emika_panda/", 1)[1]
    dst = os.path.join(ROOT, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with z.open(n) as f, open(dst, "wb") as out:
        shutil.copyfileobj(f, out)
print("extracted", len(names), "files")
