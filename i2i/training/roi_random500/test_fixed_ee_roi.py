import copy
import unittest
import numpy as np
import torch
from fixed_ee_roi import BoundROIDataset, VIEWS, ee_point, image_digest, project, rectangular_canvas


def views():
    return {n: dict(source_hw=[256,320], ee_pixels={}) for n in VIEWS}


class Tests(unittest.TestCase):
    def test_robotwin_rgb_direct_imencode_roundtrip(self):
        import cv2
        from export_hdf_preview import read_rgb
        rgb=np.zeros((32,32,3),dtype=np.uint8);rgb[:]=[220,40,10]
        ok,encoded=cv2.imencode('.jpg',rgb)
        self.assertTrue(ok)
        decoded=read_rgb([encoded.tobytes()],0)
        self.assertLess(np.abs(decoded.astype(float)-rgb).mean(),3)
        self.assertGreater(decoded[0,0,0],decoded[0,0,2])

    def test_world_pose_and_tcp_offset(self):
        np.testing.assert_allclose(ee_point([1,2,3,1,0,0,0]), [1.12,2,3])
        s = 2**-.5
        np.testing.assert_allclose(ee_point([1,2,3,s,0,0,s]), [1,2.12,3])
        np.testing.assert_allclose(ee_point([1,2,3,1,0,0,0],0), [1,2,3])

    def test_projection_3x4_4x4_and_dynamic_camera(self):
        e = np.eye(4); k = np.array([[100,0,50],[0,100,50],[0,0,1]])
        self.assertEqual(project([0,0,2],e,k), [50,50])
        e[0,3]=1
        self.assertEqual(project([0,0,2],e[:3],k), [100,50])
        self.assertIsNone(project([0,0,-2],e,k))

    def test_fixed_sides_not_gaussian_and_union(self):
        v = views(); v['head']['ee_pixels']={'left':[160,128], 'right':[160,128]}
        mask, b = rectangular_canvas(v)
        self.assertEqual(mask.sum(),96*96)
        self.assertEqual(set(mask.unique().tolist()),{0.,1.})
        v['left_wrist']['ee_pixels']={'left':[160,128]}
        mask,_ = rectangular_canvas(v)
        self.assertEqual(mask.sum(),96*96+48*48)
        self.assertEqual(mask[256:,160:].sum(),0)

    def test_border_clips_without_crossing_seam(self):
        v=views();v['left_wrist']['ee_pixels']={'left':[319,255]}
        mask,boxes=rectangular_canvas(v)
        self.assertEqual(mask[:256].sum(),0)
        self.assertEqual(mask[:,160:].sum(),0)
        self.assertLessEqual(boxes[0]['xyxy'][2],160)

    def test_invalid_points_not_clamped(self):
        v=views();v['head']['ee_pixels']={'left':[-1,40],'right':None}
        self.assertFalse(rectangular_canvas(v)[0].any())

    def test_actual_rectangle_loss_gradient_ratio(self):
        from roi_loss import weighted_flow_loss
        v=views();v['head']['ee_pixels']={'left':[160,128]}
        mask,_=rectangular_canvas(v)
        pred=torch.ones(1,1,384,320,requires_grad=True)
        def base(**kw):return ((kw['pred'][0]-kw['target'][0])**2).mean()
        loss=weighted_flow_loss(base,roi_masks=[mask[None]],strength=3.,
            pred=[pred],target=[torch.zeros_like(pred)],has_valid_tokens=True)
        loss.backward()
        self.assertAlmostEqual(loss.item(),1.,places=5)
        self.assertAlmostEqual((pred.grad[0,0,128,160]/pred.grad[0,0,0,0]).item(),4.,places=5)

    def sample(self):
        gt=torch.zeros(3,384,320,dtype=torch.uint8)
        item=dict(__url__='/data/test.mp4',target_frame_index=12,images=[gt,gt],ai_caption='original')
        v=views();v['head']['ee_pixels']={'right':[160,128]}
        record=dict(video_path='/data/test.mp4',target_frame_index=12,views=v,
            gt_chw_u8_sha256=image_digest(gt),alignment_verified=True,alignment_evidence='testfixture')
        return item,record

    def test_adapter_keeps_images_and_text(self):
        item,row=self.sample(); result=BoundROIDataset([item],[row])[0]
        self.assertIs(result['images'],item['images'])
        self.assertEqual(result['ai_caption'],item['ai_caption'])
        self.assertNotIn('_loss_roi_pair',item)
        self.assertFalse(result['_loss_roi_pair'][0].any())
        self.assertTrue(result['_loss_roi_pair'][1].any())

    def test_valid_offscreen_target_keeps_gt_and_global_loss(self):
        item,row=self.sample();row['views']=views()
        ds=BoundROIDataset([item],[row]);result=ds[0]
        self.assertIs(result['images'],item['images'])
        self.assertFalse(result['_loss_roi_pair'].any())
        self.assertEqual(ds.offscreen_targets,1)

    def test_wrong_frame_hash_and_missing_metadata_raise(self):
        item,row=self.sample();row['gt_chw_u8_sha256']='wrong'
        with self.assertRaises(ValueError):BoundROIDataset([item],[row])[0]
        with self.assertRaises(KeyError):BoundROIDataset([item],[])[0]
        row['alignment_verified']=False
        with self.assertRaises(ValueError):BoundROIDataset([item],[row])


if __name__=='__main__':unittest.main(verbosity=2)
