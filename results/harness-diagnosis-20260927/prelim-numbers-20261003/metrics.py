"""Per-run metrics shared by the paper-number scripts (read-only; no model calls).

Official USD API list prices checked 2026-10-06; rates are per million tokens
in the order uncached input, cached input, output. These are direct USD quotes,
not currency conversions. DeepSeek uses peak prices; promotional and off-peak
discounts are excluded. Historical experiment price files remain frozen.
"""
import json,statistics as st,collections
PRICE={'glm':(0.15,0.03,0.50),'deepseek':(0.30,0.006,1.20)}
PRICE_CURRENCY='USD'
PRICE_AS_OF='2026-10-06'
PRICE_SOURCES={
    'glm':'https://docs.z.ai/guides/overview/pricing',
    'deepseek':'https://api-docs.deepseek.com/quick_start/pricing',
}
def run_metrics(model,native,official):
    r=json.load(open(native)); o=json.load(open(official))
    vr=o.get('verifier_result')
    if not vr: return None
    rew=(vr.get('rewards') or {}).get('reward')
    exc=o.get('exception_info'); exc=exc.get('exception_type') if isinstance(exc,dict) else exc
    tok=collections.defaultdict(lambda: collections.Counter()); unknown_cache=0; W=0.0
    for q in r.get('requests',[]):
        p=q.get('purpose')
        if p=='compaction': W+=q.get('provider_wall_duration_seconds') or 0
        u=q.get('usage') or {}
        if not u: continue
        inp=u.get('input_tokens',0) or 0; c=u.get('cached_tokens') or 0
        if not u.get('cached_tokens_observed',True) and not c: unknown_cache+=1
        tok[p]['U']+=inp-c; tok[p]['H']+=c; tok[p]['O']+=u.get('output_tokens',0) or 0; tok[p]['n']+=1
    comps=[x for x in r.get('audit',{}).get('compactions',[]) if x.get('outcome')=='summarized']
    kept=[x['proposal'].get('retained_projection_tokens') for x in comps if x.get('proposal')]
    cyc=r.get('summary_consumption',{}).get('foreground_cycles',{}).get('cycles',[])
    steps=sum(x.get('accepted_logical_step_count',0) for x in cyc)
    post=[x.get('accepted_logical_step_count',0) for x in cyc if x.get('generation',0)>0]
    return dict(model=model,reward=rew,exception=exc,elapsed=r.get('elapsed_seconds') or 0,W=W,steps=steps,
                comps=len(comps),retrig=sum(s<=1 for s in post),kept=st.median([k for k in kept if k]) if any(kept) else None,
                tok={p:dict(v) for p,v in tok.items()},unknown_cache=unknown_cache,
                # final answer given while a background command was still running; the harness stopped it before grading
                left_running=any(f.get('error_type')=='UnfinishedCommandsError' for f in r.get('failures',[])))
def cost(m,r=None):
    pu,pc,po=PRICE[m['model']]; rr=pc/pu if r is None else r
    tot=0; summ=0
    for p,v in m['tok'].items():
        c=pu*(v.get('U',0)+rr*v.get('H',0))/1e6+po*v.get('O',0)/1e6; tot+=c
        if p=='compaction': summ+=c
    return tot,summ
def tokens(m):
    U=H=O=0
    for v in m['tok'].values(): U+=v.get('U',0); H+=v.get('H',0); O+=v.get('O',0)
    return U,H,O
def pooled(ms):
    """Token totals and steps of several runs, in the shape rstar() and step_cost() take."""
    t=[tokens(m) for m in ms]
    return tuple(sum(x[i] for x in t) for i in range(3))+(sum(m['steps'] for m in ms),)
def step_cost(model,p,r=None):
    """USD per step for pooled totals p=(U,H,O,steps), with the cached price at r times the uncached price."""
    pu,pc,po=PRICE[model]; rr=pc/pu if r is None else r; U,H,O,n=p
    return (pu*U+rr*pu*H+po*O)/1e6/n
def rstar(model,c,l):
    """Per-step crossover of C(r)=p_u U+r p_u H+p_o O: compaction totals c against L totals l."""
    pu,_,po=PRICE[model]
    (cu,ch,co),(lu,lh,lo)=[(U/n,H/n,O/n) for U,H,O,n in (c,l)]
    return -((cu-lu)+(po/pu)*(co-lo))/(ch-lh)
