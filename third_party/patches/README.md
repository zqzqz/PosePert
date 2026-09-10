# Third-party patches

`third_party/V2X-Real` needs two local changes before the V2X-Real experiments
will run. `scripts/link_local_data.sh` links an already-patched working copy, so
this only matters when cloning V2X-Real fresh.

## `v2x-real-ego-id.patch`

V2X-Real identifies vehicle agents with **negative** IDs, but upstream
`late_fusion_dataset.py` uses `ego_id = -1` as its "not found yet" sentinel and
then asserts `ego_id != -1`. A scene whose ego really is vehicle `-1` therefore
trips the assertion. The patch switches the sentinel to `None`.

```bash
cd third_party/V2X-Real
git apply ../patches/v2x-real-ego-id.patch
```

## Fusion configs

`point_pillar_intermediate_fusion_local.yaml` and
`point_pillar_intermediate_fusion_v2v.yaml` are not in upstream V2X-Real. Copy
them in:

```bash
cp third_party/patches/point_pillar_intermediate_fusion_*.yaml \
   third_party/V2X-Real/opencood/hypes_yaml/
```
