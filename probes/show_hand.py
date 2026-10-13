import os
ROOT = os.path.dirname(os.path.abspath(__file__))
s = open(os.path.join(ROOT, "panda.xml")).read()
i = s.find('name="hand"')
print(s[i-200:i+1500])
