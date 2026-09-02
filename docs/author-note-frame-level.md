# Draft note to the model author — frame-level generation

Draft for the maintainer to send; keep it factual. Nothing here is
posted automatically.

---

We have been experimenting with pushing ARTalk toward frame-level
streaming and wanted to share what we measured, and ask two questions.

**What we did.** Using the released `train_code`, we retrained both
stages with a 1 s chunk (`CLIP_LENGTH 25`, `V_PATCH_NUMS [1, 5, 25]`,
three chunks of previous context) on ~190 h of tracked data in the 106-D
layout. The codec matches or beats the released one on lip vertex error
and mean vertex distance on our held-out split, and the generator runs
in our realtime pipeline with a 1 s latency floor. We also built the
data in the 108-D layout following your note that `train_code`'s
settings are the reference; the eye channel carries real blink and gaze
motion there.

**What did not work.** A frame-by-frame model — per-frame causal audio
encoder, GRU state, one motion frame out per 40 ms in, direct regression
with velocity and acceleration losses — trains cleanly but generates
muted motion: generated-to-ground-truth velocity ratio 0.47, flat from a
quarter of training onward, while the teacher-forced losses kept
improving. The chunk models sit at 0.83-0.96 on the same measure, which
we attribute to sampling discrete tokens rather than regressing.

**What we plan.** A per-frame variant of your codec (`CLIP_LENGTH 1`,
`V_PATCH_NUMS [1]`: one 32-bit BSQ token per frame, causal by
construction) and a causal generator that samples one token per frame
with the same per-bit cross-entropy and top-p sampling as `ARTalkGen`.

**Two questions.**

1. Have you tried a per-frame or short-window causal codec, and did the
   32-bit-per-frame budget reconstruct well enough? Our 25-frame codec
   lands at ~40 bits per frame.
2. Your streaming successor is causal at 80 ms internally but still
   decodes in chunks. Is sub-chunk or per-frame decoding something you
   are pursuing? We would rather align with your direction than
   duplicate it, and we are happy to share the data builder, the eval
   scripts, and the numbers above.
