"""Train-only shared photometric jitter; preserve historical channel contracts."""
import torch

CONFIG=dict(brightness=0.3,contrast=0.4,saturation=0.5,hue=0.0,
            shared_across=['obs','future_video','goal','all_cameras'],padding_unchanged=True)

def sample_factors(generator):
    u=torch.rand(3,generator=generator).tolist()
    return tuple(1+(2*x-1)*CONFIG[k] for x,k in zip(u,('brightness','contrast','saturation')))

def jitter_canvas(canvas,mask,boxes,factors,*,order):
    assert canvas.dtype==torch.uint8 and canvas.ndim==4 and canvas.shape[0]==3
    assert order in ('rgb','bgr') and mask.shape==canvas.shape[-2:]
    result=canvas.clone();b,c,s=factors
    for y,x,h,w in boxes.values():
        valid=mask[y:y+h,x:x+w].bool()
        if not valid.any():continue
        old=canvas[:,:,y:y+h,x:x+w]
        rgb=(old[[2,1,0]] if order=='bgr' else old).float()/255
        rgb=(rgb*b).clamp(0,1)
        weights=rgb.new_tensor([.2989,.587,.114])[:,None,None,None]
        grey=(rgb*weights).sum(0,keepdim=True)
        mean=(grey*valid[None,None]).sum((-2,-1),keepdim=True)/valid.sum()
        rgb=(rgb*c+mean*(1-c)).clamp(0,1)
        grey=(rgb*weights).sum(0,keepdim=True)
        rgb=(rgb*s+grey*(1-s)).clamp(0,1)
        changed=(rgb*255).round().byte()
        if order=='bgr':changed=changed[[2,1,0]]
        result[:,:,y:y+h,x:x+w]=torch.where(valid[None,None],changed,old)
    return result

def augment_sample(sample,generator):
    factors=sample_factors(generator)
    sample['video']=jitter_canvas(sample['video'],sample['video_pixel_mask'],sample['camera_boxes'],factors,order='bgr')
    sample['goal_frame']=jitter_canvas(sample['goal_frame'],sample['goal_pixel_mask'],sample['goal_camera_boxes'],factors,order='rgb')
    return sample
