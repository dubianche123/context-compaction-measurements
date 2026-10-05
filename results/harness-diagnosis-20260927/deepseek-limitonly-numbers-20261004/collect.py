"""DeepSeek L numbers for the paper (read-only; no model calls).
Run from the repository root:
python3 results/harness-diagnosis-20260927/deepseek-limitonly-numbers-20261004/collect.py

DeepSeek L runs are measured here with the shared per-run metrics. DeepSeek S/A
and GLM rows come from the prelim snapshot (prelim-numbers-20261003/runs.json,
2026-10-03 23:08); all DeepSeek S/A runs had finished by then."""
import json,glob,os,re,sys,statistics as st
HERE=os.path.dirname(os.path.abspath(__file__))
PRELIM=os.path.join(os.path.dirname(HERE),'prelim-numbers-20261003')
sys.path.insert(0,PRELIM)
from metrics import PRICE,run_metrics,cost,tokens,pooled,step_cost,rstar
G='results/glm-repetition-tiered-20260930/work/runtime/output/harbor-jobs'
SHORT={'coq-block-bound':'Coq','interleaved-vigenere':'Vig','mp-checkpoint-consolidation':'MP','retro-console-soc':'Retro','rs-archive-clone':'RS'}
L=[]
def say(*a): L.append(' '.join(str(x) for x in a))
def per100(ms): return 100*sum(cost(m)[0] for m in ms)/sum(m['steps'] for m in ms)

dsl=[]
for d in sorted(glob.glob(f'{G}/deepseek-flash-window-sweep-20260928/limitonly-20261004/deepseek-window-limitonly-20261004-*-b001-n1m')):
    t=re.search(r'limitonly-20261004-(.+)-b001-n1m$',d).group(1)
    nat=glob.glob(d+'/*/axiom-private/result.json')[0]; off=glob.glob(d+'/*/result.json')[0]
    m=run_metrics('deepseek',nat,off); m.update(policy='L',task=t,thr=1000,rep='b001')
    m['peak']=max(((q.get('usage') or {}).get('input_tokens') or 0) for q in json.load(open(nat)).get('requests',[]))
    dsl.append(m)
snap=json.load(open(os.path.join(PRELIM,'runs.json')))
dssa=[m for m in snap if m['model']=='deepseek' and m['policy'] in ('S','A')]
glm=[m for m in snap if m['model']=='glm']

say('## DeepSeek L, one run per task (fixed price 2.00/0.04/8.00 CNY per M)')
for m in dsl:
    U,H,O=tokens(m)
    say(f"{SHORT[m['task']]:5s} reward={m['reward']} exception={m['exception']} steps={m['steps']} elapsed={m['elapsed']/60:.1f} min "
        f"compactions={m['comps']} peak prompt={m['peak']} cached={H/(U+H):.4f} cost={cost(m)[0]:.2f} CNY ({100*cost(m)[0]/m['steps']:.2f}/100 steps)")
U=sum(tokens(m)[0] for m in dsl); H=sum(tokens(m)[1] for m in dsl); O=sum(tokens(m)[2] for m in dsl); n=sum(m['steps'] for m in dsl)
pu,pc,po=PRICE['deepseek']; tot=(pu*U+pc*H+po*O)/1e6
say(f"pooled: steps={n} cost={tot:.2f} CNY, {100*tot/n:.2f}/100 steps; cached share of input {H/(U+H):.4f}; "
    f"cost shares output {po*O/1e6/tot:.2f}, cached input {pc*H/1e6/tot:.2f}, uncached input {pu*U/1e6/tot:.2f}")
say(f"pooled without Coq: {per100([m for m in dsl if m['task']!='coq-block-bound']):.2f}/100 steps")
say(f"one step at a 1M-token prompt, with L's mean uncached input ({U/n:.0f}) and output ({O/n:.0f}) per step: "
    f"{100*(pu*U/n+pc*(1e6-U/n)+po*O/n)/1e6:.2f}/100 steps")

say('\n## DeepSeek S/A (snapshot)')
for m in sorted(dssa,key=lambda m:(m['task'],m['thr'],m['policy'])):
    say(f"{SHORT[m['task']]:5s} {m['policy']}{m['thr']:<3d} reward={m['reward']} steps={m['steps']} elapsed={m['elapsed']/3600:.2f} h cost={cost(m)[0]:.2f} CNY ({100*cost(m)[0]/m['steps']:.2f}/100 steps)")
for pol in 'SA':
    xs=[m for m in dssa if m['policy']==pol]; say(f"pooled {pol}: {per100(xs):.2f}/100 steps; L/{pol} = {per100(dsl)/per100(xs):.2f}")
say(f"per run: L {min(cost(m)[0] for m in dsl):.2f}-{max(cost(m)[0] for m in dsl):.2f} CNY over {min(m['elapsed'] for m in dsl)/60:.0f}-{max(m['elapsed'] for m in dsl)/60:.0f} min; "
    f"S/A {min(cost(m)[0] for m in dssa):.2f}-{max(cost(m)[0] for m in dssa):.2f} CNY over {min(m['elapsed'] for m in dssa)/3600:.2f}-{max(m['elapsed'] for m in dssa)/3600:.2f} h")
say(f"passes: L {sum(m['reward']==1 for m in dsl)}/{len(dsl)}, S/A {sum(m['reward']==1 for m in dssa)}/{len(dssa)}")

say('\n## Per-step crossover r* against the same task\'s L (DeepSeek actual r=0.02)')
byt={m['task']:m for m in dsl}
for m in sorted(dssa,key=lambda m:(m['task'],m['thr'],m['policy'])):
    say(f"{SHORT[m['task']]:5s} {m['policy']}{m['thr']:<3d} r*={rstar('deepseek',pooled([m]),pooled([byt[m['task']]])):.3f}")

say('\n## GLM per-step r* by setting against pooled same-task L (snapshot; GLM actual r=0.29)')
rs=[]
for t in SHORT:
    lt=pooled([m for m in glm if m['policy']=='L' and m['task']==t])
    for pol,thr in sorted({(m['policy'],m['thr']) for m in glm if m['policy']!='L' and m['task']==t}):
        r=rstar('glm',pooled([m for m in glm if m['policy']==pol and m['thr']==thr and m['task']==t]),lt); rs.append(r)
        say(f"{SHORT[t]:5s} {pol}{thr:<3d} r*={r:.3f}")
say(f"GLM max r* = {max(rs):.3f}")
open(os.path.join(HERE,'output.txt'),'w').write('\n'.join(L)+'\n')
json.dump(dsl,open(os.path.join(HERE,'deepseek-l-runs.json'),'w'))
print('\n'.join(L))
