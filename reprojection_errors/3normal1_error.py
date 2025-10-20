import numpy as np
import cv2

# === Camera calibration results ===
camera_matrix = np.array([
    [3.38821744e+03, 0.00000000e+00, 9.01478839e+02],
    [0.00000000e+00, 3.35612041e+03, 5.95209942e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
], dtype=np.float32)

dist_coeffs = np.array(
    [1.37363472e+00, -6.84275187e+01, 1.46233947e-02, 6.68579741e-02, 1.36646007e+03],
    dtype=np.float32
)

# === 3D coordinates from digitizer (participant 3) ===
object_points = np.array([
    [-28.074, -34.442, -32.830],  # LPA
    [-10.058, -61.998, -33.829],  # left eye
    [-4.072, -63.890, -37.184],   # nose bridge
    [-4.051, -68.297, -31.693],   # nose tip
    [3.225, -62.993, -36.104],    # right eye
    [23.532, -34.509, -35.252],   # RPA
    [21.661, -45.563, -40.846],   # F8
    [8.265, -63.832, -45.362],    # FP2
    [-2.305, -64.465, -50.735],   # middle green
    [-9.751, -64.897, -44.166],   # FP1
    [-26.465, -46.242, -38.796],  # F7
    [-5.427, -38.773, -73.517],   # CZ
    [-10.658, -6.408, -45.221],   # O1
    [-4.993, -6.047, -45.565],    # OZ
    [0.681, -6.158, -45.815],     # O2
    [-5.427, -38.773, -73.517]    # Real CZ (duplicate)
], dtype=np.float32)

# === 2D image coordinates from labeled frames ===
image_points = np.array([
    [602, 375],  # LPA
    [552, 360],  # left eye
    [490, 375],  # nose bridge
    [490, 440],  # nose tip
    [425, 360],  # right eye
    [375, 375],  # RPA
    [775, 365],  # F8
    [425, 310],  # FP2
    [472, 252],  # middle green
    [540, 305],  # FP1
    [503, 450],  # F7
    [472, 140],  # CZ
    [385, 390],  # O1
    [420, 425],  # OZ
    [465, 445],  # O2
    [472, 140]   # Real CZ
], dtype=np.float32)

# === Compute reprojection error ===
object_points = object_points.reshape(-1, 1, 3)
image_points = image_points.reshape(-1, 1, 2)

success, rvec, tvec = cv2.solvePnP(
    object_points, image_points, camera_matrix, dist_coeffs,
    flags=cv2.SOLVEPNP_ITERATIVE
)

projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
errors = np.linalg.norm(image_points - projected_points, axis=2)
mean_error = np.mean(errors)

print("Reprojection errors per point (pixels):", np.round(errors.flatten(), 2))
print("Mean reprojection error: {:.2f} pixels".format(mean_error))
