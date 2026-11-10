"""DRS VPT: localize images in point cloud maps, and calibrate cameras to LiDARs.

    import drs_vpt
    model = drs_vpt.load_model(device="cuda")
    map_encoding = drs_vpt.encode_map(model, cloud_xyz, grid_size=0.02)
    image = drs_vpt.read_image("image.png")                            # (H, W, 3) uint8 RGB
    out = drs_vpt.localize(model, image, map_encoding, K)               # out["T_w2c"]: x_cam = R @ x_cloud + t
    calib = drs_vpt.calibrate(model, [(image, K, scan), ...])           # calib["T_cam_lidar"]
"""

from drs_vpt.hub import load_model, load_sample
from drs_vpt.inference import (calibrate, calibrate_records, calibration_record, encode_map, localize, read_image,
                               resize_to_model)

__all__ = ["load_model", "load_sample", "read_image", "resize_to_model", "encode_map", "localize", "calibrate",
           "calibration_record", "calibrate_records"]
