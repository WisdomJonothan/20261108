import sys,unittest
import torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train_ce_fusion import lambdarank_ndcg_loss
class BoundaryTests(unittest.TestCase):
 def test_outside_positive(self):
  s=torch.arange(12.,0.,-1,requires_grad=True); y=torch.zeros(12); y[-1]=1
  l=lambdarank_ndcg_loss(s,y); l.backward()
  self.assertGreater(l.item(),0);self.assertLess(s.grad[-1].item(),0)
 def test_no_positive(self):
  s=torch.arange(12.,requires_grad=True);l=lambdarank_ndcg_loss(s,torch.zeros(12));l.backward();self.assertEqual(l.item(),0)
 def test_z_invariance(self):
  s=torch.tensor([1.,3.,9.]);z=lambda x:(x-x.mean())/x.std(unbiased=False)
  self.assertTrue(torch.allclose(z(s),z(s*5+7),atol=1e-6))
if __name__=='__main__': unittest.main()
