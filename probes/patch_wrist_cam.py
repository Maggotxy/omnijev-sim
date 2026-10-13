"""Patch panda.xml: add wrist cameras to the hand body (two candidate orientations)."""
import re, os

ROOT = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(ROOT, "panda.xml")
s = open(p).read()
if "wrist" in s:
    print("already patched")
    raise SystemExit

cams = '''<camera name="wrist" pos="0 0 0.045" xyaxes="1 0 0 0 -1 0" fovy="70"/>
                      <camera name="wrist_alt" pos="0 0 0.045" xyaxes="-1 0 0 0 1 0" fovy="70"/>'''

anchor = '<geom mesh="hand_c" class="collision"/>'
assert anchor in s, "anchor not found"
s = s.replace(anchor, anchor + "\n                      " + cams, 1)
open(p, "w").write(s)
print("patched")
