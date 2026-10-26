#!/usr/bin/env python3
"""IPC contract checks: ordering, positions, shape, corrupted values and comparisons."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
from vision_batch import PR_SHA, assemble, compare, load


class BatchContract(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='m31-vision-test-')
        self.root=Path(self.tmp.name)
        # Real PR v3 serialization layout, image grid1x2 at28x56 pixels.
        data=struct.pack('<QIQB5IQBQ',3,1,0,1,2,1,0,0,1,0,0,1)
        data+=struct.pack('<2B8iB',0,0,0,0,0,0,0,0,28,56,0)
        (self.root/'image.mtmd').write_bytes(data)
        np.arange(2*6144,dtype='<f4').reshape(2,6144).tofile(self.root/'image.f32')
        self.m={'format':'m31-vision-ipc-v1','source_pr':PR_SHA,'n_embd':6144,'llm_inference':False,
                'encoded':True,'rows':4,'visual_rows':2,'chunks':[
                {'index':0,'row':0,'rows':1,'position':0,'n_pos':1,'type':'text','tokens':[41]},
                {'index':1,'row':1,'rows':2,'position':1,'n_pos':2,'type':'image','visual_row':0,
                 'metadata':'image.mtmd','embeddings':'image.f32','id':''},
                {'index':2,'row':3,'rows':1,'position':3,'n_pos':1,'type':'text','tokens':[42]}]}
        self.path=self.root/'manifest.json'
        self.save()

    def tearDown(self): self.tmp.cleanup()
    def save(self): self.path.write_text(json.dumps(self.m))

    def test_mixed_batch_keeps_absolute_positions(self):
        result=assemble(self.path,self.root/'out')
        self.assertEqual(result['visual_rows'],2)
        self.assertEqual(np.fromfile(self.root/'out/text_ids.i32',dtype='<i4').tolist(),[41,-1,-1,42])
        self.assertEqual(np.fromfile(self.root/'out/visual_rows.i32',dtype='<i4').tolist(),[-1,0,1,-1])
        self.assertEqual(np.fromfile(self.root/'out/positions.i32',dtype='<i4').tolist(),[0,1,2,3])
        self.assertEqual((self.root/'out/visual.f32').read_bytes(),(self.root/'image.f32').read_bytes())

    def test_overlapping_positions_rejected(self):
        self.m['chunks'][2]['position']=2;self.save()
        with self.assertRaisesRegex(ValueError,'continuity'):load(self.path)

    def test_visual_reordering_rejected(self):
        self.m['chunks'][1]['visual_row']=1;self.save()
        with self.assertRaisesRegex(ValueError,'reordered'):load(self.path)

    def test_bad_serialized_grid_rejected(self):
        self.m['chunks'][1]['rows']=3;self.m['chunks'][1]['n_pos']=3;self.save()
        with self.assertRaisesRegex(ValueError,'grid'):load(self.path)

    def test_nonfinite_and_short_embedding_rejected(self):
        np.array([float('nan')],dtype='<f4').tofile(self.root/'image.f32')
        with self.assertRaisesRegex(ValueError,'shape/values'):assemble(self.path,self.root/'bad')

    def test_compare_detects_difference(self):
        second=self.root/'second.json'
        m=copy.deepcopy(self.m);m['chunks'][1]['embeddings']='changed.f32'
        second.write_text(json.dumps(m))
        a=np.fromfile(self.root/'image.f32',dtype='<f4');a[0]+=1;a.tofile(self.root/'changed.f32')
        self.assertFalse(compare(self.path,second,0,0)['pass'])
        self.assertTrue(compare(self.path,self.path,0,0)['pass'])


if __name__=='__main__':unittest.main()
