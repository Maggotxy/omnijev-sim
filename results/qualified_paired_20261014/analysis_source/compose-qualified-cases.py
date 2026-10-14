"""Full-speed paired recordings annotated with retained task completion."""
import argparse,hashlib,json,sys
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jev_reflex.run import font
p=argparse.ArgumentParser();p.add_argument("root",type=Path);a=p.parse_args()
report=json.loads((a.root/"task_comparison.json").read_text());videos=[]
for pair in report["pairs"]:
    dirs=[a.root/pair["brain_dir"],a.root/pair["mixed_dir"]]
    results=[json.loads((d/"result.json").read_text()) for d in dirs]
    readers=[imageio.get_reader(str(d/"episode.mp4")) for d in dirs];counts=[r.count_frames() for r in readers]
    name=pair["case_id"].replace(":","_seed")+"_repeat"+str(pair["repeat"])
    target=a.root/("comparison_"+name+".mp4");last=[None,None];samples=[]
    with imageio.get_writer(str(target),fps=20,quality=7) as writer:
        for i in range(max(counts)):
            canvas=Image.new("RGB",(1920,800),(7,12,22));draw=ImageDraw.Draw(canvas)
            draw.text((20,8),"QUALIFIED TASK | "+pair["case_id"]+" | repeat "+str(pair["repeat"])+" | 1x wall time",font=font(25),fill=(238,244,250))
            for col,(reader,result) in enumerate(zip(readers,results)):
                metric=result["task_completion"]
                outcome=("retained completion %.2fs"%metric["retained_completion_wall_s"]) if metric["success"] else "FAILED terminal stable hold"
                label="Brain only" if col==0 else "Brain + cerebellum"
                draw.text((20+960*col,46),label+" | "+outcome,font=font(19),fill=(137,223,179) if metric["success"] else (255,173,136))
                if i<counts[col]:last[col]=reader.get_data(i)
                canvas.paste(Image.fromarray(last[col]),(960*col,80))
            writer.append_data(np.asarray(canvas))
            if i in [0,counts[0]//2,max(counts)-1]:samples.append(canvas.copy())
            if i%200==0:print(name+" "+str(i)+"/"+str(max(counts)),flush=True)
    for r in readers:r.close()
    sheet=Image.new("RGB",(1920,800*len(samples)))
    for i,im in enumerate(samples):sheet.paste(im,(0,800*i))
    sheet.save(a.root/("comparison_"+name+"_contact.png"))
    check=imageio.get_reader(str(target));actual=check.count_frames();meta=check.get_meta_data();check.close()
    assert actual==max(counts) and meta["fps"]==20 and list(meta["size"])==[1920,800]
    videos.append({"file":target.name,"frames":actual,"fps":20,"size":[1920,800],"sha256":hashlib.sha256(target.read_bytes()).hexdigest(),**pair})
    (a.root/"videos.json").write_text(json.dumps(videos,indent=2))
print(json.dumps({"videos":len(videos)}),flush=True)
