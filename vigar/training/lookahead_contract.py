"""Match subgoal planner's non-final segment tail redirect; never skip input samples."""
import math

def select_stage(starts, ends, frame, ratio):
    if not 0 <= ratio <= 1:raise ValueError(ratio)
    if len(starts)!=len(ends) or not starts:raise ValueError('Invalid stages')
    for i,(start,end) in enumerate(zip(starts,ends)):
        if start <= frame < end:
            tail=math.ceil((end-start)*ratio)
            return min(i+1,len(ends)-1) if tail and frame>=end-tail else i
    raise ValueError(f'Frame outside stage intervals: {frame}')

def policy_case(cases,frame,ratio=.15):
    starts=[0]+[int(c['end_frame_exclusive']) for c in cases[:-1]]
    ends=[int(c['end_frame_exclusive']) for c in cases]
    return cases[select_stage(starts,ends,frame,ratio)]
