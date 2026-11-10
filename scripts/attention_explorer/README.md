# Attention explorer data

The attention explorer (`attention/` in the [project page repo](https://github.com/ori-drs/drs-vpt-pages)) shows, for one query image per scene, where each of the 12 global attention blocks sends a pixel's
attention in the map, and a per-block PCA of the map-token features. These scripts regenerate its data. Run them from
the repository root with `drs_vpt` importable.

```bash
D=../drs-vpt-pages/attention/data     # a checkout of ori-drs/drs-vpt-pages next to this repo

# 7-Scenes stairs (Hub sample)
python scripts/attention_explorer/export_attention.py --sample 7scenes --out $D/stairs

# KITTI: VP2P test samples 2233, 1116 and 5583, localized as in the benchmark (0.02 m voxels, no point cap)
python scripts/attention_explorer/make_kitti_samples.py --data /data/kitti_vp2p_eval_data --out samples 2233 1116 5583
python scripts/attention_explorer/export_attention.py --sample samples/kitti_002233 --out $D/kitti-junction
python scripts/attention_explorer/export_attention.py --sample samples/kitti_001116 --out $D/kitti-street
python scripts/attention_explorer/export_attention.py --sample samples/kitti_005583 --out $D/kitti-truck

# Trafalgar Square: a MegaDepth map (WAI format), localized with the training point cap. The query is a freely
# licensed photo from Wikimedia Commons ("National Gallery Facade" by Nathan Stazicker, CC BY-SA 3.0, the 1024 px
# version), not a MegaDepth photo, so it has no ground-truth pose and the model predicts its intrinsics.
python scripts/attention_explorer/make_megadepth_sample.py --scene /data/wai/megadepth/0001_0 --out samples/trafalgar \
    --query 56 --map-views 25 --rotate-seed 10
python scripts/attention_explorer/export_attention.py --sample samples/trafalgar --image National_Gallery_Facade.jpg \
    --predict-k --out $D/trafalgar --max-points 38912
```

MegaDepth photos are Flickr images under their owners' licences (often all rights reserved), so the page shows only
query photos whose licence allows it.

The page lists its scenes in `SCENES` in `attention/index.html` of the page repo; a new scene needs an entry there and a button.
