import numpy as np
import pandas as pd
import cv2
import os
import glob

# Load calibration results
camera_matrix = np.array([
    [3.38821744e+03, 0.00000000e+00, 9.01478839e+02],
    [0.00000000e+00, 3.35612041e+03, 5.95209942e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
], dtype=np.float32)

dist_coeffs = np.array([1.37363472, -68.4275187, 0.0146233947, 0.0668579741, 1366.46007], dtype=np.float32)


# Extract 10 evenly spaced frames from a video 
def extract_frames(video_name, num_frames=10, target_w=1280):
    video_path = f"data/videos/{video_name}.mov"
    output_dir = f"data/frames/{video_name}"
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

    saved = 0
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if w > target_w:
            new_h = int(h * (target_w / w))
            frame = cv2.resize(frame, (target_w, new_h))
        frame_name = f"{video_name}_frame_{idx:06d}.jpg"
        cv2.imwrite(os.path.join(output_dir, frame_name), frame)
        saved += 1

    cap.release()
    print(f"Saved {saved} evenly spaced frames for {video_name} to {output_dir}")


# Run solvePnP for a single image 
def run_solvepnp(video_name, frame_img_path, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)

    model_3d_csv = f"data/points/{video_name}.csv"
    pts2d_name = os.path.splitext(os.path.basename(frame_img_path))[0]
    image_2d_csv = f"data/points/{video_name}_{pts2d_name}_2d.csv"

    if not os.path.exists(model_3d_csv):
        raise FileNotFoundError(f"3D model file not found: {model_3d_csv}")
    if not os.path.exists(image_2d_csv):
        raise FileNotFoundError(f"2D points file not found: {image_2d_csv}")

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
    out_img = os.path.join(out_dir, f"{video_name}_{base}_projection.jpg")
    out_npz = os.path.join(out_dir, f"{video_name}_{base}_pose.npz")

    cv2.imwrite(out_img, vis)
    np.savez(out_npz, rvec=rvec, tvec=tvec, error=float(err))

    print(f"Pose estimation succeeded for {video_name} | {base}")
    print(f"Mean Reprojection Error: {err:.4f} pixels")
    return err


# Main interface
def main():
    video_name = input("Enter video base name (e.g., 15normal1): ").strip()

    print("\nSelect mode:")
    print("1) Extract 10 frames from video")
    print("2) Run solvePnP on a single image")
    print("3) Run solvePnP on all frames for this video")
    mode = input("Enter 1, 2, or 3: ").strip()

    if mode == "1":
        extract_frames(video_name)

    elif mode == "2":
        frame_img_path = input("Enter image path (jpg/png): ").strip()
        run_solvepnp(video_name, frame_img_path)

    elif mode == "3":
        frames_folder = f"data/frames/{video_name}"
        image_paths = sorted(glob.glob(os.path.join(frames_folder, "*.jpg")))

        if not image_paths:
            print(f"No images found in {frames_folder}")
            return

        print(f"Found {len(image_paths)} frames. Running solvePnP on all...\n")
        errors = []
        for img_path in image_paths:
            try:
                err = run_solvepnp(video_name, img_path)
                errors.append(err)
            except Exception as e:
                print("Error processing", img_path, ":", e)

        if errors:
            avg_err = np.mean(errors)
            print(f"\nAverage Reprojection Error: {avg_err:.4f} pixels")
            pd.DataFrame({"frame": image_paths, "error": errors}).to_csv(
                f"outputs/{video_name}_summary.csv", index=False
            )
            print(f"Results saved to outputs/{video_name}_summary.csv")
        else:
            print("\nNo valid results produced.")

    else:
        print("Invalid mode selected.")


if __name__ == "__main__":
    main()

