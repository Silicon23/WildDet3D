# Third-party submodule patches

Records of local changes to the `third_party/` submodule **checkouts**, kept
here because the submodule pins themselves must stay at their upstream commits.

> **Read the status line on each patch below before doing anything.** Unlike the
> patches in the sibling `open-vocab-3d-tracker` repo — which are *applied* and
> in some cases auto-applied at runtime by an `ensure_*_patched()` helper — a
> patch here may be a historical record that is deliberately **NOT** applied.
> Applying one without reading its section will make the submodule dirty for no
> reason.

Check whether a patch is currently applied to a checkout:

```bash
# exit 0 => already applied      exit non-zero => not applied
git -C third_party/<submodule> apply --check --reverse \
    third_party/patches/<name>.patch
```

Apply / revert by hand:

```bash
git -C third_party/<submodule> apply           third_party/patches/<name>.patch  # apply
git -C third_party/<submodule> apply --reverse third_party/patches/<name>.patch  # revert
```

## sam3_level_start_index_blackwell.patch

**Status: UNAPPLIED — and that is intentional. Do not apply it routinely.**
`third_party/sam3` is clean at its pin `159490b`, and should stay that way
unless the symptom below actually appears.

One line in `sam3/model/encoder.py`, `TransformerEncoder.forward`, where the
FPN level offsets are computed:

```python
-  spatial_shapes.prod(1).cumsum(0)[:-1],
+  (spatial_shapes[:, 0] * spatial_shapes[:, 1]).cumsum(0)[:-1],
```

It replaces a `prod` reduction over `dim=1` of an `int64 [num_levels, 2]` tensor
with the equivalent explicit `H * W` multiply. The two forms are numerically
identical — verified `torch.equal` on the real FPN shapes
(`[[252,252],[126,126],[63,63]] -> [63504, 15876, 3969]`) — so this is not a
correctness fix. It only avoids *calling* `Tensor.prod`.

**Why it is kept.** The edit appeared in the working tree with mtime
**2026-07-12**, which falls inside the window when this box was an **NVIDIA B300
(sm_103, Blackwell Ultra)** — first observed 2026-07-08, replaced by an H100 on
2026-07-26. No agent claims authorship and no note was left, but the shape fits
the other Blackwell gaps hit in that window (see the `b300-torch-blackwell`
project memory: torch 2.11.0+cu128 for sm_100 kernels, `triton==3.0.0`,
`xformers==0.0.35 --no-deps`, `TORCH_CUDNN_SDPA_ENABLED=0` for cuDNN's SDPA
backend). A missing or broken `prod` reduction kernel for sm_103 would produce
exactly this workaround.

**Why it is not applied.** On the current H100 the upstream `prod(1)` form runs
fine, so the workaround buys nothing on the hardware we run today, and a
permanently-dirty file inside a *nested* submodule is close to invisible state —
it cost three agents time during a repo audit precisely because nothing surfaced
it. Reverted 2026-08-21 in favour of this record.

**Try this first if Blackwell comes back.** If WildDet3D's image encoder starts
failing inside `TransformerEncoder.forward` at the `level_start_index`
computation — on a Blackwell/sm_103 device, plausibly a "no kernel image is
available" or an internal reduction error out of `Tensor.prod` — apply this
patch before debugging anything else:

```bash
git -C third_party/sam3 apply third_party/patches/sam3_level_start_index_blackwell.patch
```

That leaves `third_party/sam3` dirty against its pin, which is expected while
the workaround is needed; revert it once the box is back on hardware where
upstream `prod` works.

**Caveat on the evidence.** The equivalence test above was run on an H100, so it
establishes that the two forms agree and that upstream works *there*. It cannot
disprove a Blackwell-specific `prod` failure, which is the whole reason this
record exists rather than the diff simply being discarded. Note also that the
hardware under this project has changed without notice once already, so "the
B300 is gone" is a statement about today.

Consumed by Track C: `wilddet3d/track_c/feature_extractor.py` builds the frozen
WildDet3D stack, which runs this encoder on every precomputed frame.
