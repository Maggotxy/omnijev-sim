import os
ROOT = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(ROOT, "panda.xml")
s = open(p).read()
s = s.replace('pos="0 0 -0.06"', 'pos="0 0 0.045"')
open(p, "w").write(s)
print("moved cams to 0 0 0.045:", s.count('pos="0 0 0.045"'))
