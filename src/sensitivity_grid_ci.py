"""Per-setting bootstrap intervals and correlation p-values for the floor x decay grid.


Writes results/sensitivity/grid_ci.csv. Uses 1,000 attacks per scenario and 1,000
bootstrap resamples of whole networks per (floor, decay) setting.
"""
import sys, json, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
import simulate as S
from simulate import prepare, simulate
from scoring import build_table, P_CEIL
from topology import load_topology, topology_names
from paths import CONFIGS, CATALOG, EPSS, SENSITIVITY
from sensitivity import share, add_logit
pc=pd.read_csv(os.path.join(os.path.dirname(SENSITIVITY),'network_risk.csv')).set_index('network').pc
TR=1000; out=[]
for fl in (0.02,0.05,0.10):
  for dc in (0.3,0.5,0.7):
    S.DECAY=dc
    tbl=build_table(CATALOG,EPSS,floor=fl,ceil=P_CEIL,transform="linear")[0]
    pk=sorted(tbl); ix={s:i for i,s in enumerate(pk)}
    q=np.array([(tbl[s]-fl)/(P_CEIL-fl) for s in pk]); rows=[]
    for ti,t in enumerate(topology_names()):
      G,nodes,_=load_topology(t); A,D,idx=prepare(G,nodes)
      for d in (0.2,0.4,0.6,0.8):
        for s in range(1,6):
          cn=json.load(open(os.path.join(CONFIGS,f"{t}_d{d}_s{s}.json")))["nodes"]
          sw=[cn[v]["software"] for v in nodes]; p=np.array([tbl[x] for x in sw])
          e=idx[[v for v in nodes if cn[v]["is_entry_point"]][0]]
          f=simulate(A,D,p,e,TR,mode="decay",seed=1000*ti+100*int(d*10)+s,shared=(np.array([ix[x] for x in sw]),q,fl,P_CEIL))["frac"].mean()
          rows.append((t,len(nodes),d,p.mean(),f))
    df=pd.DataFrame(rows,columns=["topo_id","n_nodes","density","mp","f"]); df=add_logit(df,"f")
    pt=share(df)
    rng=np.random.default_rng(0); groups={t:g for t,g in df.groupby("topo_id")}; tops=list(groups); bs=[]
    for _ in range(1000):
        parts=[]
        for i,t in enumerate(rng.choice(tops,len(tops))):
            g=groups[t].copy(); g["topo_id"]=f"{t}_{i}"; parts.append(g)
        bs.append(share(pd.concat(parts)))
    lo,hi=np.percentile(bs,[2.5,97.5])
    g=df.groupby(["topo_id","density"]); dx=df.mp-g.mp.transform("mean"); dyl=df.y-g.y.transform("mean")
    r,pr=pearsonr(dx,dyl); nr=df.groupby("topo_id").f.mean(); rho,prho=spearmanr(pc.loc[nr.index],nr)
    out.append(dict(floor=fl,decay=dc,topology=round(pt,3),ci_low=round(lo,3),ci_high=round(hi,3),r=round(r,3),p_r=pr,rho=round(rho,3),p_rho=prho))
    print(out[-1],flush=True)
pd.DataFrame(out).to_csv(os.path.join(SENSITIVITY,'grid_ci.csv'),index=False,float_format='%.6g')