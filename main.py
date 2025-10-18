import os
import cv2
import numpy as np
import pandas as pd
import glob
import gdown

# ====================== CAMERA CALIBRATION ======================
camera_matrix = np.array([
    [3.38821744e+03, 0.00000000e+00, 9.01478839e+02],
    [0.00000000e+00, 3.35612041e+03, 5.95209942e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
], dtype=np.float32)

dist_coeffs = np.array([1.37363472, -68.4275187, 0.0146233947, 0.0668579741, 1366.46007], dtype=np.float32)

# ====================== STEP 1: DOWNLOAD VIDEO FROM GOOGLE DRIVE ======================
def download_video_from_drive(file_id, output_dir="data/videos"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{file_id}.mov")
    if os.path.exists(output_path):
        print(f"Video already exists: {output_path}")
        return output_path
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading video from Drive: {file_id} ...")
    gdown.download(url, output_path, quiet=False)
    print(f"Saved to {output_path}")
    return output_path

# ====================== STEP 2: EXTRACT FRAMES ======================
def extract_frames(video_path, subject_id, output_dir="data/frames", total_frames=10):
    subject_dir = os.path.join(output_dir, str(subject_id))
    os.makedirs(subject_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // total_frames)
    saved = 0

    for i in range(0, frame_count, step):
        if saved >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        filename = os.path.join(subject_dir, f"{subject_id}_frame_{saved+1:03d}.jpg")
        cv2.imwrite(filename, frame)
        saved += 1

    cap.release()
    print(f"Extracted {saved} frames for subject {subject_id} to {subject_dir}")

# ====================== STEP 3: RUN SOLVEPNP ======================
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

# ====================== STEP 4: MAIN MENU ======================
def main():
    print("\n=== MAIN MENU ===")
    print("1) Download videos from Google Drive and extract frames")
    print("2) Run solvePnP on a single image")
    print("3) Run solvePnP on all images in a folder")
    mode = input("Enter 1, 2, or 3: ").strip()

    if mode == "1":
        n = int(input("How many videos to process? "))
        for _ in range(n):
            drive_link = input("Paste Google Drive link or ID: ").strip()
            subject_id = input("Enter subject number (e.g., 15): ").strip()
            file_id = drive_link.split('/')[-2] if 'drive.google.com' in drive_link else drive_link
            video_path = download_video_from_drive(file_id)
            extract_frames(video_path, subject_id)
            os.remove(video_path)
            print(f"Deleted video file {video_path} after frame extraction.\n")

    elif mode == "2":
        model_3d_csv = "data/points/model_3d.csv"
        frame_img_path = input("Enter image path (jpg/png): ").strip()
        points_2d_csv = input("Enter 2D points file (name,x,y): ").strip()
        run_solvepnp(model_3d_csv, points_2d_csv, frame_img_path)

    elif mode == "3":
        model_3d_csv = "data/points/model_3d.csv"
        points_2d_csv = input("Enter 2D points file (name,x,y): ").strip()
        frames_folder = "data/frames"
        image_paths = sorted(glob.glob(os.path.join(frames_folder, "*.jpg")))
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
        print("Invalid option selected.")


if __name__ == "__main__":
    main()
