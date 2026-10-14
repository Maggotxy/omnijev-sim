"""Paired retained task times; failed runs have no assigned completion time."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument("root",type=Path);a=p.parse_args()
rows=json.loads((a.root/"summary.json").read_text());cases=list(dict.fromkeys(r["case_id"] for r in rows))
fig,axes=plt.subplots(1,len(cases),figsize=(15,4.6),sharex=True,sharey=True,squeeze=False)
colors={"vlm":"#3969ae","vlm_omni":"#229978"}
for ax,case in zip(axes[0],cases):
    group=[r for r in rows if r["case_id"]==case]
    for repeat in [1,2,3]:
        points=[]
        for mode,dy in [("vlm",.13),("vlm_omni",-.13)]:
            row=next(r for r in group if r["repeat"]==repeat and r["mode"]==mode);metric=row["task_completion"];y=4-repeat+dy
            if metric["success"]:
                x=metric["retained_completion_wall_s"];points.append((x,y))
                ax.scatter(x,y,s=60,color=colors[mode],zorder=3,label=mode if repeat==1 else None)
                ax.text(x+.4,y+.04 if dy>0 else y-.12,f"{x:.2f}s",fontsize=9,color=colors[mode])
            else:ax.text(35.5,y-.04,"FAIL",fontsize=10,color=colors[mode],fontweight="bold")
        if len(points)==2:ax.plot(*zip(*points),color="#b0b7bf",linewidth=1.5,zorder=1)
    ax.set_title(case);ax.set_xlim(0,40);ax.set_ylim(.5,3.6)
    ax.set_yticks([1,2,3],["Repeat 3","Repeat 2","Repeat 1"]);ax.grid(axis="x",alpha=.2)
    ax.set_xlabel("Final retained completion (wall seconds)")
fig.suptitle("Fresh paired tests after brain-only qualification | blue: brain | green: brain + cerebellum",fontsize=13)
fig.text(.5,.01,"Completion requires 40 stable control steps and retention through the end; FAIL has no completion-time value.",ha="center",fontsize=10)
fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(a.root/"task_completion_comparison.png",dpi=160);plt.close(fig)
print("saved task_completion_comparison.png")
