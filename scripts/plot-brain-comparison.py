"""Standard plots of saved timestamps and measured robot motion."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument("root",type=Path);args=p.parse_args();root=args.root
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":160})
metrics=json.loads((root/"selection_metrics.json").read_text());summary=json.loads((root/"summary.json").read_text())
fig,axes=plt.subplots(1,3,figsize=(13,4.2),constrained_layout=True)
labels=["Brain only","Brain + cerebellum"]
for ax,field,title in ((axes[0],"selected_capture_to_motor_p50_s","Selected command age (median, s)"),(axes[1],"changed_capture_to_motor_p50_s","Changed command age (median, s)")):
    values=[metrics["groups"][m][field] for m in ("vlm","vlm_omni")]
    ax.bar(labels,values,color=["#3968b3","#23986f"])
    for i,v in enumerate(values):ax.text(i,v+.12,f"{v:.2f}",ha="center")
    ax.set_title(title);ax.set_ylim(0,6);ax.grid(axis="y",alpha=.2)
x=np.arange(2);matches=[metrics["groups"][m]["matched_events"] for m in ("vlm","vlm_omni")]
axes[2].bar(labels,matches,color=["#3968b3","#23986f"]);axes[2].set_title("Matched new commands / 12 events")
axes[2].set_ylim(0,12);axes[2].grid(axis="y",alpha=.2)
for i,v in enumerate(matches):axes[2].text(i,v+.2,f"{v}/12",ha="center")
fig.suptitle("30 episodes, same visual brain; all matched responses are target corrections",fontsize=12)
fig.savefig(root/"comparison_overview.png");plt.close(fig)
for seed in (541,577,613):
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True,constrained_layout=True)
    for mode,label,color in (("vlm","Brain only","#3968b3"),("vlm_omni","Brain + cerebellum","#23986f")):
        d=root/f"target_jump_seed{seed}_{mode}"
        states=[json.loads(s) for s in (d/"states.jsonl").read_text().splitlines()]
        result=json.loads((d/"result.json").read_text());event=result["reactions"][0]
        t=np.array([s["wall_s"] for s in states]);sim=np.array([s["sim_s"] for s in states])
        axes[0].plot(t,[s["eef_position_m"][1] for s in states],label=label,color=color)
        axes[1].plot(t,[np.linalg.norm(s["measured_eef_velocity_m_s"]) for s in states],color=color,label=label,alpha=.8)
        if mode=="vlm":
            target=event["start_y"]+(event["end_y"]-event["start_y"])*np.clip((sim-event["onset_sim_s"])/.7,0,1)
            axes[0].plot(t,target,label="Target Y (evaluation only)",color="black",ls="--")
            onset_wall=float(t[np.searchsorted(sim,event["onset_sim_s"])])
            for ax in axes:ax.axvline(onset_wall,color="gray",ls=":",label="Event onset" if ax is axes[0] else None)
        actual=next(e for e in metrics["events"] if e["mode"]==mode and e["scenario"]=="target_jump" and e["seed"]==seed)
        first=actual["first"]
        if first:
            for ax in axes:ax.axvline(first["selected_wall_s"],color=color,ls="--",alpha=.7)
            axes[0].text(first["selected_wall_s"]+.2,.02,f"{first['source']}:{first['action']}",color=color,rotation=90,va="bottom",fontsize=8)
    axes[0].set_ylabel("Measured end-effector Y (m)");axes[0].legend(loc="upper left",ncol=2,fontsize=8)
    axes[1].set_ylabel("Measured speed (m/s)");axes[1].set_xlabel("Actual wall-clock seconds");axes[1].set_xlim(0,26)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle(f"Target displacement, seed {seed}: observed motion, no time compression")
    fig.savefig(root/f"target_jump_seed{seed}_motion.png");plt.close(fig)
print(json.dumps({"plots":4,"root":str(root)}))
