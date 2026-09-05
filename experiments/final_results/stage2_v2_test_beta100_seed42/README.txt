Final Stage2-v2 TEST evaluation

Selected model:
Qwen/Qwen2.5-3B-Instruct
LoRA-adapted contextual reranker
beta = 1.0
optimizer step = 1254

Checkpoint SHA256:
ed4a90bd4b9d4c7fcca6f41ef6ee54038a2f548b3a2217ac97697b438ca994a4

Evaluation fingerprint:
d2fcbad11a343a2784a8fdb881a4f465bc2448537d269670ebaaa6a77ce4f652

Final TEST results:

RRF:
R@1  = 3.437660%
R@10 = 17.034377%
R@50 = 34.992304%
MRR  = 7.711971%

LoRA-adapted Qwen2.5:
R@1  = 5.618266%
R@10 = 21.292971%
R@50 = 34.992304%
MRR  = 10.462417%

Final artifact SHA256:
instances = a7d19065f419726c2aa121fe0e08bff0bd2ee4dac13c202e9a94d5a1199e465a
summary   = 99883ea3021831ccf6ec0a274a25b4c29f03e4a86d3379a5c318021aad25fd97
manifest  = 2ead4232ff1723bbb65b14868e415f90f8f3a43d2fca587133589fb8fe1a6346

The TEST split was not used for model selection.
The selected beta=1.0 configuration was chosen using TRAIN-derived DEV MRR.
