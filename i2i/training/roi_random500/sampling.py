"""One shared 50-task model; task x domain balanced without per-domain routing."""
from bisect import bisect_left
from collections import defaultdict
import random

class UnifiedIndex:
    def __init__(self,episodes,samples,metadata,seed=42):
        self.samples,self.seed=samples,seed;self.by_episode={};groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        for e,episode in enumerate(episodes):
            meta=metadata[int(episode.episode_id)];task,domain=meta['task'],meta['source_config']
            assert domain in ('seg1_clean','rand250');self.by_episode[e]=(task,domain)
            for phase,(start,end,mistake,*_) in enumerate(episode.segments):
                if mistake or end-start<2:continue
                lo,hi=bisect_left(samples,(e,start)),bisect_left(samples,(e,end-1))
                if lo<hi:groups[task][phase][domain].append((e,start,end,lo,hi))
        self.groups={t:{p:dict(d) for p,d in sorted(phases.items())} for t,phases in sorted(groups.items())}
        self.tasks=sorted(self.groups)
        assert len(self.tasks)==50,'Unified experiment must cover all 50 tasks'
        missing=[(t,p,k) for t,phases in self.groups.items() for p,d in phases.items() for k in ('seg1_clean','rand250') if not d.get(k)]
        if missing:raise ValueError('Missing task/phase/domain training examples: '+repr(missing))

    def resolve(self,index):
        index=int(index);rng=random.Random(self.seed+index*1000003)
        task=self.tasks[(index//10)%len(self.tasks)]
        domain='seg1_clean' if index%10==0 else 'rand250'
        phase=rng.choice(list(self.groups[task]));e,start,end,lo,hi=rng.choice(self.groups[task][phase][domain])
        b=rng.random();low,high=(0,.25) if b<.6 else ((.25,.75) if b<.9 else (.75,.95))
        frame=min(end-2,start+int(rng.uniform(low,high)*(end-start-1)))
        resolved=min(hi-1,max(lo,bisect_left(self.samples,(e,frame))))
        assert self.samples[resolved][0]==e and self.samples[resolved][1]<end-1
        return resolved
