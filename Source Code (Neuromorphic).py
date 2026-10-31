"""Reproducible synthetic admission-control experiment, not an SNN benchmark.
Python 3 + NumPy + SciPy. Run: python experiment.py
Outputs: synthetic_runs.csv, paired_statistics.csv, analytic_grid.csv.
All cost units are normalized events; no hardware joules or task accuracy.
"""
import csv
from pathlib import Path
import numpy as np
from scipy import stats
POLICIES=['All events','Random q=exp(-0.4)','Static theta=0.4','Queue only','Budget only','Joint controller']
T=1200; WARM=200; SERVICE=80; BUDGET=60; SEEDS=30

def run(seed, keep_trace=False):
    rng=np.random.default_rng(seed)
    rr=np.random.default_rng(10000+seed)
    q=np.zeros(6); dual=np.zeros(6); traces=[]
    total=np.zeros((6,3)) # accepted, accepted utility, offered utility
    hist=[[] for _ in POLICIES]
    for t in range(T):
        rate=55+65*np.sin(2*np.pi*t/200)**2
        scores=rng.exponential(1.0,rng.poisson(rate))
        mask=[np.ones(len(scores),bool),rr.random(len(scores))<np.exp(-.4),
              scores>=.4,scores>=.15+q[3]/40,
              scores>=.15+dual[4],scores>=.15+dual[5]+q[5]/40]
        accepted=np.array([m.sum() for m in mask],float)
        q=np.maximum(q+accepted-SERVICE,0)
        for j in (4,5): dual[j]=max(0,dual[j]+.0005*(accepted[j]-BUDGET))
        if t>=WARM:
            for j,m in enumerate(mask):
                total[j]+=[accepted[j],scores[m].sum(),scores.sum()]
                hist[j].append(q[j])
        if keep_trace: traces.append([t,rate,*q,dual[5],accepted[5]])
    rows=[]
    for j,name in enumerate(POLICIES):
        a,u,s=total[j]
        rows.append(dict(seed=seed,policy=name,accepted_per_window=a/(T-WARM),
                         retained_utility=u/s,queue_mean=np.mean(hist[j]),
                         queue_p95=np.percentile(hist[j],95),queue_max=max(hist[j])))
    return rows,np.asarray(traces)

def produce():
    rows=[]
    for seed in range(SEEDS): rows.extend(run(seed)[0])
    contrasts=[]
    for metric,a,b in [('retained_utility',2,1),('queue_mean',5,4),('accepted_per_window',5,3)]:
        av=np.array([r[metric] for r in rows if r['policy']==POLICIES[a]])
        bv=np.array([r[metric] for r in rows if r['policy']==POLICIES[b]])
        d=av-bv; se=stats.sem(d); ci=stats.t.ppf(.975,SEEDS-1)*se
        contrasts.append(dict(metric=metric,contrast=POLICIES[a]+' minus '+POLICIES[b],
                         difference=d.mean(),ci_low=d.mean()-ci,ci_high=d.mean()+ci,
                         t=stats.ttest_rel(av,bv).statistic,p=stats.ttest_rel(av,bv).pvalue,
                         dz=d.mean()/d.std(ddof=1)))
    order=np.argsort([x['p'] for x in contrasts]); last=0
    for rank,j in enumerate(order):
        last=max(last,min(1,(len(order)-rank)*contrasts[j]['p']))
        contrasts[j]['p_holm']=last
    analytic=[]
    for p in [1,4,16,64,256,1024]:
        for rho in [.05,.2,.5,.8]:
            for theta in [0,.2,.4,.5,.8,1.2]:
                q=np.exp(-theta); R=.04+rho*q*(1.1+.02*np.log2(p))
                analytic.append(dict(cores=p,density=rho,theta=theta,q=q,
                                     utility_retained=(1+theta)*q,energy_ratio=R))
    return rows,contrasts,analytic

def save_csv(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':
    rows,contrasts,analytic=produce()
    for name,data in [('synthetic_runs.csv',rows),('paired_statistics.csv',contrasts),('analytic_grid.csv',analytic)]: save_csv(name,data)
    for p in POLICIES:
        rr=[r for r in rows if r['policy']==p]
        print(p,{k:round(np.mean([r[k] for r in rr]),5) for k in list(rr[0])[2:]})
    print(contrasts)
