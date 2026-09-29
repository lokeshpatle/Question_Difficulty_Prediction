from __future__ import annotations

def extract(doc, negation_terms, clause_heads):
    import math
    if doc is None: return {'F09':math.nan,'F10':math.nan,'F11':math.nan,'F12':math.nan,'F13':math.nan}
    try:
        f09=sum(1 for t in doc if t.dep_.lower() in {x.lower() for x in clause_heads})
        chunks=[len(list(c)) for c in doc.noun_chunks]
        f10=(sum(chunks)/len(chunks)) if chunks else 0.0
        max_depth=0; mdds=[]
        for sent in doc.sents:
            root_i=sent.root.i
            for tok in sent:
                depth=0; cur=tok
                while cur.head != cur and depth<100:
                    depth+=1; cur=cur.head
                max_depth=max(max_depth,depth)
            nonroots=[t for t in sent if t.head!=t]
            if len(nonroots)>=1 and len(list(sent))>=2:
                mdds.append(sum(abs(t.i-t.head.i) for t in nonroots)/(len(list(sent))-1))
        f13=len({t.i for t in doc if t.dep_.lower()=='neg' or t.text.casefold() in negation_terms})
        return {'F09':f09,'F10':f10,'F11':max_depth,'F12':(sum(mdds)/len(mdds) if mdds else math.nan),'F13':f13}
    except Exception:
        return {k:math.nan for k in ('F09','F10','F11','F12','F13')}
