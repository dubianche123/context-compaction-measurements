"""Preliminary paper numbers from finished runs (read-only; no model calls).
Run from the repository root: python3 results/harness-diagnosis-20260927/prelim-numbers-20261003/collect.py [--from-data]
--from-data reprices the frozen runs.json without collecting runs or rewriting that snapshot."""
import argparse,json,glob,os,re,statistics as st,collections,datetime
OUT=os.path.dirname(os.path.abspath(__file__))
from metrics import PRICE,PRICE_AS_OF,run_metrics,cost
parser=argparse.ArgumentParser()
parser.add_argument("--from-data",action="store_true",help="Report from frozen runs.json; do not collect or rewrite runs")
args=parser.parse_args()
G='results/glm-repetition-tiered-20260930/work/runtime/output/harbor-jobs'
TASKS=['coq-block-bound','interleaved-vigenere','mp-checkpoint-consolidation','retro-console-soc','rs-archive-clone']
def short(t): return {'coq-block-bound':'Coq','interleaved-vigenere':'Vig','mp-checkpoint-consolidation':'MP','retro-console-soc':'Retro','rs-archive-clone':'RS'}[t]
runs=[]
def add(model,policy,task,thr,rep,native,official):
    m=run_metrics(model,native,official)
    if m: m.update(policy=policy,task=task,thr=thr,rep=rep,native=native,official=official); runs.append(m)  # paths feed wait_metrics.py
if args.from_data:
    runs=json.load(open(os.path.join(OUT,"runs.json")))
else:
    # GLM S/A, all finished canonical runs (b001-b003)
    for d in glob.glob(f'{G}/glm-window-sweep-20260927/main/long-011-formal-summary12288/window-main-long-011-formal-summary12288-*-b00[123]-[sa]*'):
        m=re.search(r'summary12288-(.+)-(b00\d)-([sa])(\d+)$',d)
        if not m: continue
        t,b,a,w=m.groups(); nat=glob.glob(d+'/*/axiom-private/result.json'); off=glob.glob(d+'/*/result.json')
        if nat and off: add('glm',a.upper(),t,int(w),b,nat[0],off[0])
    # GLM K
    for c in json.load(open('monitoring/glm-skeep-endpoint-summary-20261003.json'))['cells']:
        t,b,w=re.search(r'skeep-001-(.+)-(b00\d)-k(\d+)',c['job']).groups(); add('glm','K',t,int(w),b,c['native_result'],c['official_result'])
    # GLM L (no threshold compaction): b001 runs live in three result roots
    for d in glob.glob('results/*/work/runtime/output/harbor-jobs/glm-window-sweep-20260927/main/long-011/n1m-deadline5400-b001/window-main-long-011-d5400-*-b001-n1m')+\
             glob.glob(f'{G}/glm-window-sweep-20260927/main/long-011/n1m-b00[23]-summary12288/window-main-long-011-summary12288-*-b00[23]-n1m'):
        t,b=re.search(r'(?:d5400|summary12288)-(.+)-(b00\d)-n1m$',d).groups(); nat=glob.glob(d+'/*/axiom-private/result.json'); off=glob.glob(d+'/*/result.json')
        if nat and off: add('glm','L',t,1000,b,nat[0],off[0])
    # DeepSeek S/A pairs
    for d in glob.glob(f'{G}/deepseek-flash-window-sweep-20260928/main/long-011-formal-summary12288/deepseek-window-main-long-011-formal-summary12288-*-b001-[sa]*'):
        t,a,w=re.search(r'summary12288-(.+)-b001-([sa])(\d+)$',d).groups()
        if t=='coq-block-bound': continue  # branch source only
        nat=glob.glob(d+'/*/axiom-private/result.json'); off=glob.glob(d+'/*/result.json')
        if nat and off: add('deepseek',a.upper(),t,int(w),'b001',nat[0],off[0])
    # DeepSeek L (no threshold compaction): one run per task, after the S/A pairs
    for d in glob.glob(f'{G}/deepseek-flash-window-sweep-20260928/limitonly-20261004/deepseek-window-limitonly-20261004-*-b001-n1m'):
        t=re.search(r'limitonly-20261004-(.+)-b001-n1m$',d).group(1); nat=glob.glob(d+'/*/axiom-private/result.json'); off=glob.glob(d+'/*/result.json')
        if nat and off: add('deepseek','L',t,1000,'b001',nat[0],off[0])
L=[]
def say(*a): L.append(' '.join(str(x) for x in a))
say('generated',datetime.datetime.now().isoformat(timespec='minutes'),'runs',len(runs),collections.Counter((m['model'],m['policy']) for m in runs))
say('prices: official USD per million tokens, checked',PRICE_AS_OF,PRICE,'(DeepSeek peak; no promotional/off-peak discounts)')
# 1. waiting share in S runs
say('\n## 1. Share of elapsed time spent generating summaries (S runs)')
for model in ('glm','deepseek'):
    for thr in (32,64,128):
        xs=[m for m in runs if m['model']==model and m['policy']=='S' and m['thr']==thr and m['elapsed']]
        if not xs: continue
        fr=sorted(100*m['W']/m['elapsed'] for m in xs)
        say(f'{model} S{thr}: n={len(xs)} median {st.median(fr):.1f}% min {fr[0]:.1f}% max {fr[-1]:.1f}%  |',', '.join(f"{short(m['task'])}{m['rep'][-1]}:{100*m['W']/m['elapsed']:.1f}%" for m in sorted(xs,key=lambda m:(m['task'],m['rep']))))
for model in ('glm','deepseek'):
    ds=[]
    for m in runs:
        pass
say('summary call duration: see per-model medians below')
# 2. repetition spread in elapsed time
say('\n## 2. Elapsed time across repetitions of the same setting (GLM S/A/K)')
grp=collections.defaultdict(list)
for m in runs:
    if m['model']=='glm' and m['policy'] in ('S','A','K'): grp[(m['policy'],m['task'],m['thr'])].append(m['elapsed']/3600)
ratios=[(max(v)/min(v),k,v) for k,v in grp.items() if len(v)>=2 and min(v)>0]
ratios.sort()
say('settings with >=2 runs:',len(ratios),'median max/min ratio %.1f'%st.median(r for r,_,_ in ratios),'largest:',[(k,[round(x,2) for x in v]) for r,k,v in ratios[-3:]])
# 3. task success
say('\n## 3. Official task success (reward=1) by model/policy/threshold')
for model in ('glm','deepseek'):
    for pol in ('S','A','K','L'):
        for thr in (32,64,128,1000):
            xs=[m for m in runs if m['model']==model and m['policy']==pol and m['thr']==thr]
            if xs: say(f'{model} {pol}{thr if thr<1000 else ""}: {sum(m["reward"]==1 for m in xs)}/{len(xs)}  by task',{short(t):f'{sum(m["reward"]==1 for m in xs if m["task"]==t)}/{sum(1 for m in xs if m["task"]==t)}' for t in TASKS if any(m['task']==t for m in xs)})
# paired GLM S/A discordance
pairs=collections.defaultdict(dict)
for m in runs:
    if m['model']=='glm' and m['policy'] in ('S','A'): pairs[(m['task'],m['thr'],m['rep'])][m['policy']]=m
done=[p for p in pairs.values() if 'S' in p and 'A' in p]
say('GLM complete S/A pairs:',len(done),'both pass',sum(p['S']['reward']==1 and p['A']['reward']==1 for p in done),
    'S only',sum(p['S']['reward']==1 and p['A']['reward']!=1 for p in done),'A only',sum(p['A']['reward']==1 and p['S']['reward']!=1 for p in done),
    'neither',sum(p['S']['reward']!=1 and p['A']['reward']!=1 for p in done))
# 4. cost per 100 steps and summarization share
say('\n## 4. Cost per 100 agent steps (fixed list prices) and summarization share')
for model in ('glm','deepseek'):
    for pol in ('S','A','K','L'):
        for thr in (32,64,128,1000):
            xs=[m for m in runs if m['model']==model and m['policy']==pol and m['thr']==thr and m['steps']]
            if not xs: continue
            tot=sum(cost(m)[0] for m in xs); summ=sum(cost(m)[1] for m in xs); steps=sum(m['steps'] for m in xs)
            line=f'{model} {pol}{thr if thr<1000 else ""}: n={len(xs)} USD/100 steps {100*tot/steps:.2f}  summarization share {100*summ/tot:.1f}%'
            if model=='glm':
                t2=sum(cost(m,0.02)[0] for m in xs); line+=f'  | repriced at r=0.02: {100*t2/steps:.2f}'
                comps=sum(m['comps'] for m in xs); line+=f'  | compactions/100 steps {100*comps/steps:.2f}, re-trigger<=1 step {sum(m["retrig"] for m in xs)}/{comps}'
            say(line)
# per-task L vs S/A at both price ratios (GLM)
say(f'\nGLM per task, USD per 100 steps at r={PRICE["glm"][1]/PRICE["glm"][0]:.2g} / r=0.02:')
for t in TASKS:
    row=[]
    for pol,thr in (('L',1000),('S',128),('A',128),('S',64),('A',64),('S',32),('A',32)):
        xs=[m for m in runs if m['model']=='glm' and m['policy']==pol and m['thr']==thr and m['task']==t and m['steps']]
        if not xs: continue
        st_=sum(m['steps'] for m in xs); row.append(f'{pol}{"" if thr==1000 else thr} {100*sum(cost(m)[0] for m in xs)/st_:.2f}/{100*sum(cost(m,0.02)[0] for m in xs)/st_:.2f}')
    say(' ',short(t),' | '.join(row))
# 5. K mechanism
say('\n## 5. Kept steps (K): median kept tokens per switch')
for thr in (32,64):
    ks=[m['kept'] for m in runs if m['policy']=='K' and m['thr']==thr and m['kept']]
    if ks: say(f'K{thr}: median of per-run medians {st.median(ks):.0f} tokens, range {min(ks):.0f}-{max(ks):.0f}')
# summary call durations
for model in ('glm','deepseek'):
    pass
# 6. final answer with a background command still running (the harness stopped it before grading)
say('\n## 6. Final answer with a background command still running')
xs=[m for m in runs if m['left_running']]
say('runs',len(xs),'passed',sum(m['reward']==1 for m in xs),'by model/policy',dict(collections.Counter(f"{m['model']} {m['policy']}" for m in xs)),
    'by task',dict(collections.Counter(short(m['task']) for m in xs)))
open(os.path.join(OUT,'prelim-output.txt'),'w').write('\n'.join(L)+'\n')
if not args.from_data:
    json.dump(runs,open(os.path.join(OUT,'runs.json'),'w'))
print('\n'.join(L))
