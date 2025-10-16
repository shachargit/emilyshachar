import numpy as np
import pandas as pd
import cv2
import os
import glob

# ====== שלב 1: תוצאות הכיול ======
camera_matrix = np.array([
    [3.38821744e+03, 0.00000000e+00, 9.01478839e+02],
    [0.00000000e+00, 3.35612041e+03, 5.95209942e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
], dtype=np.float32)

dist_coeffs = np.array([1.37363472, -68.4275187, 0.0146233947, 0.0668579741, 1366.46007], dtype=np.float32)

# ====== שלב 2: חילוץ פריימים מהווידאו ======
def extract_frames(video_path, output_dir="data/frames", step=30, max_frames=400, target_w=1280):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"לא ניתן לפתוח את הווידאו {video_path}")
    i = saved = 0
    while True:
        ok, frame = cap.read()
        if not ok or saved >= max_frames:
            break
        if i % step == 0:
            h, w = frame.shape[:2]
            if w > target_w:
                new_h = int(h * (target_w / w))
                frame = cv2.resize(frame, (target_w, new_h))
            cv2.imwrite(os.path.join(output_dir, f"frame_{i:06d}.jpg"), frame)
            saved += 1
        i += 1
    cap.release()
    print(f"שמרתי {saved} פריימים בתיקייה {output_dir}")

# ====== שלב 3: solvePnP בודד ======
def run_solvepnp(model_3d_csv, image_2d_csv, frame_img_path, out_dir="data/frames"):
    os.makedirs(out_dir, exist_ok=True)
    model = pd.read_csv(model_3d_csv)
    pts2d = pd.read_csv(image_2d_csv)
    merged = pd.merge(model, pts2d, on="name", how="inner")

    if len(merged) < 6:
        raise ValueError("צריך לפחות 6 נקודות תואמות בין 3D ל-2D")

    object_points = merged[["X","Y","Z"]].to_numpy(np.float32)
    image_points  = merged[["x","y"]].to_numpy(np.float32)

    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    if not ok:
        raise RuntimeError("פתרון solvePnP נכשל")

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1,2)

    frame = cv2.imread(frame_img_path)
    vis = frame.copy()
    for (x,y) in projected:
        cv2.circle(vis, (int(x), int(y)), 6, (0,255,0), -1)
    for (x,y) in image_points:
        cv2.circle(vis, (int(x), int(y)), 4, (0,0,255), -1)

    err = cv2.norm(image_points, projected, cv2.NORM_L2) / len(object_points)
    base = os.path.splitext(os.path.basename(frame_img_path))[0]
    out_img = os.path.join(out_dir, f"{base}_projection.jpg")
    out_npz = os.path.join(out_dir, f"{base}_pose.npz")

    cv2.imwrite(out_img, vis)
    np.savez(out_npz, rvec=rvec, tvec=tvec, error=float(err))

    print("✅ Pose estimation succeeded!")
    print(f"📸 {base} | 📉 Mean Reprojection Error: {err:.4f}px")
    return err

# ====== שלב 4: ממשק ראשי ======
def main():
    print("בחרי מצב:", "1) לחלץ פריימים מווידאו", "2) להריץ solvePnP על תמונה", "3) להריץ solvePnP על כל התמונות בתיקייה", sep="\n")
    mode = input("הקלידי 1, 2 או 3: ").strip()

    if mode == "1":
        video_path = input("נתיב וידאו: ").strip()
        extract_frames(video_path)

    elif mode == "2":
        model_3d_csv = "data/points/model_3d.csv"
        frame_img_path = input("נתיב לתמונה (jpg/png): ").strip()
        points_2d_csv = input("נתיב לקובץ נקודות 2D (name,x,y): ").strip()
        run_solvepnp(model_3d_csv, points_2d_csv, frame_img_path)

    elif mode == "3":
        model_3d_csv = "data/points/model_3d.csv"
        points_2d_csv = input("נתיב לקובץ נקודות 2D (name,x,y): ").strip()
        frames_folder = "data/frames"
        image_paths = sorted(glob.glob(os.path.join(frames_folder, "*.jpg")))

        if not image_paths:
            print("❌ לא נמצאו תמונות בתיקייה data/frames")
            return

        print(f"נמצאו {len(image_paths)} תמונות. מריצה על כולן...\n")
        errors = []
        for img_path in image_paths:
            try:
                err = run_solvepnp(model_3d_csv, points_2d_csv, img_path)
                errors.append(err)
            except Exception as e:
                print("⚠️ שגיאה בתמונה", img_path, ":", e)

        if errors:
            print("\n📊 ממוצע שגיאה כולל:", np.mean(errors))
        else:
            print("\nלא התקבלו תוצאות תקינות.")

    else:
        print("בחרת מצב לא חוקי")

if __name__ == "__main__":
    main()
