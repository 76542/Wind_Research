import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb

LAND="#ece9e0"; TITLE="#1a3a5f"
STATE={"Gujarat":"#d62728","Maharashtra":"#ff7f0e","Goa":"#bcbd22","Karnataka":"#2ca02c",
       "Kerala":"#1f77b4","Tamil Nadu":"#9467bd","Andhra Pradesh":"#e377c2","Odisha":"#8c564b"}
RED="#c0392b"; AMB="#cf8a17"; GRN="#1e8449"
def tint(c,f=0.80):
    r,g,b=to_rgb(c); return (r+(1-r)*f,g+(1-g)*f,b+(1-b)*f)
def qcol(r):
    return RED if r<0.15 else (AMB if r<0.5 else GRN)

fig,ax=plt.subplots(figsize=(13.5,6.3)); fig.patch.set_facecolor("white")
ax.set_facecolor(LAND); ax.set_xlim(0,100); ax.set_ylim(11,86); ax.axis("off")
BW,BH=17,15.5
def box(cx,cy,name,r2,rmse,pts,tag=None,dashed=False):
    col=STATE[name]
    ax.add_patch(FancyBboxPatch((cx-BW/2,cy-BH/2),BW,BH,
        boxstyle="round,pad=0.02,rounding_size=2",linewidth=2.0,
        edgecolor=col,facecolor=tint(col),
        linestyle=(0,(5,3)) if dashed else "solid",zorder=3))
    ax.text(cx,cy+3.7,name,ha="center",va="center",fontsize=12.5,fontweight="bold",color="#222",zorder=4)
    ax.text(cx,cy-0.3,f"R\u00b2 {r2:.2f}  \u00b7  RMSE {rmse:.2f}",ha="center",va="center",fontsize=10.5,color="#333",zorder=4)
    sub=f"{pts} pts" if tag is None else f"{pts} pts \u00b7 {tag}"
    ax.text(cx,cy-4.3,sub,ha="center",va="center",fontsize=9.3,color="#555",zorder=4)
def arrow(p0,p1,r2,lx,ly,ha="center"):
    c=qcol(r2)
    ax.add_patch(FancyArrowPatch(p0,p1,arrowstyle="-|>",mutation_scale=15,
        linewidth=2.1,color=c,shrinkA=1,shrinkB=1,zorder=2))
    ax.text(lx,ly,f"{r2:+.2f}",ha=ha,va="center",fontsize=10.5,fontweight="bold",color=c,zorder=5,
        bbox=dict(boxstyle="round,pad=0.18",facecolor="white",edgecolor=c,linewidth=0.8))

# coords
GJ=(9,50); MH=(30,75); KA=(57,75); KL=(84,75); GOA=(30,55); TN=(30,25); AP=(57,25); OD=(84,25)
box(*GJ,"Gujarat",0.76,1.332,218,tag="home")
box(*MH,"Maharashtra",0.748,1.328,115); box(*KA,"Karnataka",0.784,1.083,47); box(*KL,"Kerala",0.694,1.288,71)
box(*GOA,"Goa",0.714,1.555,12,tag="transfer",dashed=True)
box(*TN,"Tamil Nadu",0.674,1.715,93); box(*AP,"Andhra Pradesh",0.676,1.513,101); box(*OD,"Odisha",0.681,1.673,64)

# arrows (zero-shot R2 of parent on child, before FT)
arrow((17.5,53),(21.5,71),0.40,15.5,65)         # GJ->MH
arrow((17.5,47),(21.5,29),-0.85,15.5,35)        # GJ->TN  (cross-ocean)
arrow((38.5,75),(48.5,75),0.33,43.5,79)         # MH->KA
arrow((65.5,75),(75.5,75),0.02,70.5,79)         # KA->KL  (collapse)
arrow((30,67.5),(30,62.7),0.71,34.5,65.1,ha="left")   # MH->Goa
arrow((38.5,25),(48.5,25),-0.15,43.5,29)        # TN->AP
arrow((65.5,25),(75.5,25),-1.03,70.5,29)        # AP->OD

ax.text(9,71,"West coast",ha="center",fontsize=12,fontweight="bold",color="#555")
ax.text(9,29,"East coast",ha="center",fontsize=12,fontweight="bold",color="#555")

fig.suptitle("Transfer Learning Chains: Zero-Shot Transfer vs Fine-Tuned Skill",
    fontsize=15,fontweight="bold",color=TITLE,y=0.975)
ax.set_title("Arrow = zero-shot R\u00b2 (parent model on child, before fine-tuning)   |   "
    "Box = test-set R\u00b2 / RMSE (m/s)",fontsize=11.5,color=TITLE,pad=12)

leg=[Line2D([0],[0],color=RED,lw=2.6,label="R\u00b2 < 0.15  (transfer fails)"),
     Line2D([0],[0],color=AMB,lw=2.6,label="0.15 \u2013 0.50  (weak)"),
     Line2D([0],[0],color=GRN,lw=2.6,label="\u2265 0.50  (usable as-is)")]
ax.legend(handles=leg,loc="upper center",ncol=3,frameon=True,fontsize=9.6,
    title="Zero-shot transfer skill",title_fontsize=9.5,
    bbox_to_anchor=(0.5,-0.05),edgecolor="#999")

plt.tight_layout(rect=[0,0.02,1,0.95])
plt.savefig("outputs/figure3_tl_chain.png",dpi=220,bbox_inches="tight",facecolor="white")
print("ok")