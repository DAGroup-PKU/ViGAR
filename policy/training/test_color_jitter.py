import unittest
import torch
from color_jitter import jitter_canvas,augment_sample

class TestJitter(unittest.TestCase):
    def test_shared_rgb_bgr_temporal_and_padding(self):
        rgb=torch.randint(20,180,(3,1,12,16),generator=torch.Generator().manual_seed(9),dtype=torch.uint8)
        mask=torch.zeros(12,16,dtype=torch.bool);mask[2:10,2:14]=True
        rgb[:,:,~mask]=0; boxes={'head':[0,0,12,16]}
        x={'video':rgb[[2,1,0]].repeat(1,3,1,1),'goal_frame':rgb.clone(),'video_pixel_mask':mask,'goal_pixel_mask':mask,'camera_boxes':boxes,'goal_camera_boxes':boxes,'action':torch.randn(4,49)}
        action=x['action'].clone();state=torch.random.get_rng_state().clone()
        y=augment_sample(x,torch.Generator().manual_seed(22))
        self.assertTrue(torch.equal(y['video'][[2,1,0],:1],y['goal_frame']))
        self.assertTrue(torch.equal(y['video'][:,0],y['video'][:,2]))
        self.assertFalse(y['video'][:,:,~mask].any())
        self.assertTrue(torch.equal(action,y['action']))
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        self.assertFalse(torch.equal(rgb,y['goal_frame']))
    def test_identity(self):
        x=torch.randint(0,256,(3,3,8,8),dtype=torch.uint8);m=torch.ones(8,8,dtype=torch.bool)
        self.assertTrue(torch.equal(x,jitter_canvas(x,m,{'head':[0,0,8,8]},(1,1,1),order='rgb')))

if __name__=='__main__':unittest.main()
