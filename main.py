import numpy as np
import pandas as pd
import cv2
import os
import glob

# ====== Step 1: Load calibration results ======
camera_matrix = np.array([
    [3.38821744e+03, 0.00000000e+00, 9.01478839e+02],
    [0.00000000e+00, 3.35612041e+03, 5.95209942e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
], dtype=np.float32)

dist_coeffs = np.array([1.37363472, -68.4275187, 0.0146233947, 0.0668579741, 1366.46007], dtype=np.float32)


# ====== Step 2: Extract 10 frames from local video ======
def extract_frames(video_path, output_dir="data/frames", total_frames=10, target_w=1280):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // total_frames, 1)

    name = os.path.splitext(os.path.basename(video_path))[0]
    subject_id = ''.join([c for c in name if c.isdigit()])
    subject_folder = os.path.join(output_dir, subject_id or name)
    os.makedirs(subject_folder, exist_ok=True)

    i = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok or saved >= total_frames:
            break
        if i % step == 0:
            h, w = frame.shape[:2]
            if w > target_w:
                new_h = int(h * (target_w / w))
                frame = cv2.resize(frame, (target_w, new_h))
            out_name = os.path.join(subject_folder, f"{name}_frame_{saved+1:02d}.jpg")
            cv2.imwrite(out_name, frame)
            saved += 1
        i += 1
    cap.release()
    print(f"Saved {saved} frames to {subject_folder}")


# ====== Step 3: Run solvePnP for one image ======
def run_solvepnp(model_3d_csv, image_2d_csv, frame_img_path, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    model = pd.read_csv(model_3d_csv)
    pts2d = pd.read_csv(image_2d_csv)
    merged = pd.merge(model, pts2d, on="name", how="inner")

    if len(merged) < 6:
        raise ValueError("At least 6 matching points are required between 3D and 2D sets.")

    object_points = merged[["X", "Y", "Z"]].to_numpy(np.float32)
    image_points = merged[["x", "y"]].to_numpy(np.float32)

    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    if not ok:
        raise RuntimeError("solvePnP failed to find a solution.")

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2)

    frame = cv2.imread(frame_img_path)
    vis = frame.copy()
    for (x, y) in projected:
        cv2.circle(vis, (int(x), int(y)), 6, (0, 255, 0), -1)
    for (x, y) in image_points:
        cv2.circle(vis, (int(x), int(y)), 4, (0, 0, 255), -1)

    err = cv2.norm(image_points, projected, cv2.NORM_L2) / len(object_points)
    base = os.path.splitext(os.path.basename(frame_img_path))[0]
    out_img = os.path.join(out_dir, f"{base}_projection.jpg")
    out_npz = os.path.join(out_dir, f"{base}_pose.npz")

    cv2.imwrite(out_img, vis)
    np.savez(out_npz, rvec=rvec, tvec=tvec, error=float(err))

    print(f"Pose estimation succeeded for {base}")
    print(f"Mean Reprojection Error: {err:.4f} pixels")
    return err


# ====== Step 4: Main interface ======
def main():
    print("Select mode:")
    print("1) Extract 10 frames from local video")
    print("2) Run solvePnP on a single image")
    print("3) Run solvePnP on all images in data/frames")

    mode = input("Enter 1, 2, or 3: ").strip()

    if mode == "1":
        video_path = input("Enter local video path (e.g. data/videos/15normal1.mov): ").strip()
        extract_frames(video_path)

    elif mode == "2":
        model_3d_csv = "data/points/model_3d.csv"
        frame_img_path = input("Enter image path (jpg/png): ").strip()
        points_2d_csv = input("Enter 2D points file (name,x,y): ").strip()
        run_solvepnp(model_3d_csv, points_2d_csv, frame_img_path)

    elif mode == "3":
        model_3d_csv = "data/points/model_3d.csv"
        points_2d_csv = input("Enter 2D points file (name,x,y): ").strip()
        frames_folder = "data/frames"
        image_paths = sorted(glob.glob(os.path.join(frames_folder, "**/*.jpg"), recursive=True))

        if not image_paths:
            print("No images found in data/frames")
            return

        print(f"Found {len(image_paths)} images. Running solvePnP on all...\n")
        errors = []
        for img_path in image_paths:
            try:
                err = run_solvepnp(model_3d_csv, points_2d_csv, img_path)
                errors.append(err)
            except Exception as e:
                print("Error processing", img_path, ":", e)

        if errors:
            print("\nAverage Reprojection Error:", np.mean(errors))
        else:
            print("\nNo valid results produced.")

    else:
        print("Invalid mode selected.")


if __name__ == "__main__":
    main()
