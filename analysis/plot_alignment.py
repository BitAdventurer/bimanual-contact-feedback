"""Plot the success of applying a sister branch's follow-up on the actual branch."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=Path, default=Path("runs/alignment.json"))
parser.add_argument("--output", type=Path, default=Path("runs/alignment.pdf"))
args = parser.parse_args()
data = json.loads(args.input.read_text())
labels = ["L big", "L small", "Hold", "R small", "R big"]
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8, "pdf.fonttype":42})
fig, axes = plt.subplots(1,2,figsize=(6.9,2.92))
fig.subplots_adjust(left=.10, right=.98, bottom=.27, top=.88, wspace=.19)
for ax, cohort, title in zip(axes, ("material","fresh"),("a   Held-out materials","b   Newly acquired grasps")):
    values=np.array(data[cohort]["response_matrix"])*100
    ax.imshow(values,cmap="cividis",vmin=0,vmax=100)
    for i in range(5):
        ax.add_patch(Rectangle((i-.48,i-.48),.96,.96, fill=False, linewidth=1.35, edgecolor="#e26b3a"))
        for j in range(5):
            ax.text(j,i,f"{values[i,j]:.0f}",va="center",ha="center",fontsize=7.6,color="white" if values[i,j]<48 else "#172028")
    ax.set(xticks=range(5),yticks=range(5),xticklabels=labels,yticklabels=labels)
    ax.tick_params(length=0)
    ax.set_title(title,fontsize=9,loc="left",pad=7)
axes[0].set_ylabel("Actual pin motion")
fig.text(.52,.12,"Pin motion that supplied the policy's observation",ha="center",fontsize=8)
fig.text(.52,.027,"Cell entries are percentages. Outlines mark observations from the actual motion.",ha="center",fontsize=7)
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output,bbox_inches="tight",pad_inches=.035)
plt.close(fig)
