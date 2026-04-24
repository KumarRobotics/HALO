import numpy as np

def estimate_scale(predictions, H_world2cam_priors, prev_scale):
    refined = dict(predictions)

    extrinsic_list = predictions['extrinsic'].copy()
    N = len(extrinsic_list)

    if N > len(H_world2cam_priors):
        N = len(H_world2cam_priors)

    if N == 0:
        return

    # Compute closed-form scale to fit predicted w2c to priors
    raw = np.array(predictions.get('extrinsic', []))  # shape (N,3,4) or (N,4,4)
    if raw.ndim == 3 and raw.shape[1:] == (3, 4):
        N0 = raw.shape[0]
        bottom = np.tile(np.array([0, 0, 0, 1]), (N0, 1))  # (N,4)
        bottom = bottom[:, None, :]                        # (N,1,4)
        raw = np.concatenate((raw, bottom), axis=1)        # (N,4,4)
    # now raw is (N,4,4)
    w2c_pred = [raw[i] for i in range(raw.shape[0])]
    t_pred = np.stack([H[:3,3] for H in w2c_pred], axis=0)
    t_pred = t_pred[:N]  # ignore loop closure frames
    t_ext  = np.stack([H[:3,3] for H in H_world2cam_priors], axis=0)
    if t_pred.shape[0] > t_ext.shape[0]:
        t_pred = t_pred[:t_ext.shape[0]]
    dists_pred = np.linalg.norm(t_pred[1:] - t_pred[:-1], axis=1)
    dists_ext  = np.linalg.norm(t_ext[1:] - t_ext[:-1], axis=1)
    s = np.mean(dists_ext / dists_pred) if len(dists_pred) > 0 else 1.0
    # print(f"Computed scale: {s:.4f} (prev: {prev_scale})")

    # exponential filter on scale
    if prev_scale is not None:
        alpha = 0.1
        s = alpha * s + (1 - alpha) * prev_scale

    # store computed scale for pointcloud scaling
    refined['scale'] = s

    # store gps translation and predicted rotation
    fused_extrinsics = []
    for i in range(N):
        H_prior = H_world2cam_priors[i]
        H_pred  = w2c_pred[i]
        R_pred  = H_pred[:3,:3]
        t_prior = H_prior[:3,3]
        H_fused = np.eye(4)
        H_fused[:3,:3] = R_pred
        H_fused[:3,3]  = t_prior
        fused_extrinsics.append(H_fused)
    refined['fused_extrinsics'] = np.array(fused_extrinsics)

    scaled_extrinsics = []
    for i in range(len(w2c_pred)):
        H = w2c_pred[i]
        R = H[:3,:3]
        t = H[:3,3]
        H_scaled = np.eye(4)
        H_scaled[:3,:3] = R
        H_scaled[:3,3] = t * s
        scaled_extrinsics.append(H_scaled)
    refined['scaled_extrinsics'] = np.array(scaled_extrinsics)  # (N,4,4)

    # scale depth maps
    refined['scaled_depth'] = predictions['depth'] * s

    return refined
