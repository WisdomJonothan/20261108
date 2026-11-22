import numpy as np, torch, importlib.util
from transformers import AutoTokenizer, AutoModel
spec = importlib.util.spec_from_file_location("dq", "scripts/dense_qwen3.py")
dq = importlib.util.module_from_spec(spec); spec.loader.exec_module(dq)
M="/homeC/wujiaheng/models-qwen3-emb06"
tok = AutoTokenizer.from_pretrained(M, padding_side="left")
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModel.from_pretrained(M, dtype=torch.float16).to("cuda").eval()
ids, texts = dq.load_docs("bitcoin")
sub = [texts[i] for i in range(4)]
def enc_single(t):
    e = tok([t], truncation=True, max_length=512, return_tensors="pt").to("cuda")
    with torch.no_grad(): h = model(**e).last_hidden_state[0, -1].float()
    return torch.nn.functional.normalize(h, dim=-1).cpu().numpy()
vs = [enc_single(t) for t in sub]
print("single norms", [float(np.linalg.norm(v)) for v in vs])
for i in range(4):
    for j in range(i+1,4):
        print(f"cos {i}-{j}", float(vs[i] @ vs[j]))
b = dq.embed(model, tok, sub, 4, 512, False)
print("batch norms", [float(np.linalg.norm(v)) for v in b])
print("batch vs single cos", [float(b[i] @ vs[i]) for i in range(4)])
e = tok(sub, padding=True, truncation=True, max_length=512, return_tensors="pt")
print("pad side", tok.padding_side, "shape", tuple(e["input_ids"].shape))
print("last ids", e["input_ids"][:, -1].tolist(), "len", e["attention_mask"].sum(1).tolist())
