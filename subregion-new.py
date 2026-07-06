import gc
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler

os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", message=".*force_all_finite.*")

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["axes.linewidth"] = 1.5


def evaluate_k(features, n_runs, k):
    best_ch_score = -np.inf
    best_silhouette = -1

    for _ in range(n_runs):
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=2048,
            n_init=50,
            init="k-means++",
            random_state=np.random.randint(1, 10000),
        ).fit(features)

        current_ch = calinski_harabasz_score(features, kmeans.labels_)
        if current_ch > best_ch_score:
            best_ch_score = current_ch
            best_silhouette = silhouette_score(features, kmeans.labels_)

    return k, best_ch_score, best_silhouette


def auto_kmeans_parallel(features, max_k=15, min_k=2, n_runs=5, n_jobs=4):
    if len(features) > 5e5:
        np.random.seed(42)
        sample_idx = np.random.choice(len(features), 500_000, replace=False)
        features = features[sample_idx]

    k_values = list(range(min_k, max_k + 1))
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(evaluate_k)(features, n_runs, k) for k in k_values
    )

    results.sort(key=lambda x: x[0])
    ch_scores = [item[1] for item in results]
    silhouettes = [item[2] for item in results]

    print("CH scores:")
    print(ch_scores)
    print("Silhouette scores:")
    print(silhouettes)

    return k_values, ch_scores, silhouettes


def plot_evaluation_curves(fig_path, img_name, mask_name, k_values, ch_scores, silhouettes):
    best_ch_k = k_values[int(np.argmax(ch_scores))]
    best_silhouette_k = k_values[int(np.argmax(silhouettes))]

    plt.figure(figsize=(14, 6.5))

    plt.subplot(1, 2, 1)
    plt.plot(k_values, ch_scores, "b-o", linewidth=2.5, markersize=8)
    plt.axvline(best_ch_k, color="r", linestyle="--", linewidth=2, alpha=0.8)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Calinski-Harabasz Score")
    plt.title(f"CH Peak at K={best_ch_k}")
    plt.tick_params(axis="both", labelsize=16, width=1.5, length=6)
    plt.grid(True, linestyle="--", alpha=0.35)

    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouettes, "g-s", linewidth=2.5, markersize=8)
    plt.axvline(best_silhouette_k, color="r", linestyle="--", linewidth=2, alpha=0.8)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title(f"Silhouette Peak at K={best_silhouette_k}")
    plt.tick_params(axis="both", labelsize=16, width=1.5, length=6)
    plt.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f"{img_name}_{mask_name}.png"), dpi=300)
    plt.close()


def build_features(image_in_path, image_out_path, mask_path, smooth_sigma=1.0):
    image_in = sitk.ReadImage(image_in_path)
    image_out = sitk.ReadImage(image_out_path)
    mask = sitk.ReadImage(mask_path)

    if image_in.GetSize() != mask.GetSize() or image_out.GetSize() != mask.GetSize():
        print(
            f"Size mismatch: in {image_in.GetSize()}, out {image_out.GetSize()}, mask {mask.GetSize()}"
        )
        return None

    mask_array = sitk.GetArrayFromImage(mask)
    if not np.any(mask_array):
        print(f"Empty mask: {mask_path}")
        return None

    smoothed_image_in = sitk.SmoothingRecursiveGaussian(image_in, sigma=smooth_sigma)
    smoothed_image_out = sitk.SmoothingRecursiveGaussian(image_out, sigma=smooth_sigma)
    image_in_array = sitk.GetArrayFromImage(smoothed_image_in) * mask_array
    image_out_array = sitk.GetArrayFromImage(smoothed_image_out) * mask_array

    z_coords, y_coords, x_coords = np.where(mask_array > 0)
    if len(z_coords) == 0:
        print("No valid voxels found in mask")
        return None

    voxel_in_values = image_in_array[z_coords, y_coords, x_coords].reshape(-1, 1)
    voxel_out_values = image_out_array[z_coords, y_coords, x_coords].reshape(-1, 1)
    spatial_coords = np.column_stack((x_coords, y_coords, z_coords)).astype(float)
    physical_coords = spatial_coords * image_in.GetSpacing()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_coords = scaler.fit_transform(physical_coords)
    scaled_coords = scaled_coords * [0.01, 0.01, 0.02]

    features = np.column_stack(
        (
            scaled_coords,
            voxel_in_values,
            voxel_in_values ** 2,
            voxel_out_values,
            voxel_out_values ** 2,
        )
    )

    del image_in, image_out, mask
    del smoothed_image_in, smoothed_image_out
    del image_in_array, image_out_array, mask_array
    gc.collect()
    return features


def process_case(fig_path, img_name, mask_name, image_in_path, image_out_path, mask_path, smooth_sigma=1.0):
    features = build_features(image_in_path, image_out_path, mask_path, smooth_sigma=smooth_sigma)
    if features is None:
        return False

    try:
        k_values, ch_scores, silhouettes = auto_kmeans_parallel(
            features=features,
            max_k=15,
            min_k=2,
            n_runs=5,
            n_jobs=4,
        )
        plot_evaluation_curves(fig_path, img_name, mask_name, k_values, ch_scores, silhouettes)
    except Exception as exc:
        print(f"Evaluation failed for {img_name}_{mask_name}: {exc}")
        return False
    finally:
        del features
        gc.collect()

    return True


image_in_path = r"E:\liuzhou_breastcancer\figure-res-0\washIn\Benign1.nii"
image_out_path = r"E:\liuzhou_breastcancer\figure-res-0\washOut\Benign1.nii"
mask_path = r"E:\DCESummary_2019-202004\benign-DCE-221subs\part1-175subs+part2\1\Untitled.nii.gz"
datapath = r"E:\liuzhou_breastcancer\datas_clear_final.csv"
fig_path = r"E:\liuzhou_breastcancer\class_rank1"

datalist = pd.read_csv(datapath)
error_records = []

if error_records:
    pd.DataFrame(error_records).to_csv("shape_mismatch_errors.csv", index=False)
    print("Saved shape_mismatch_errors.csv")

for datalist0 in range(datalist.shape[0]):
    if datalist.loc[datalist0, "grade"] == 0:
        folder_path = r"E:\liuzhou_breastcancer\figure-res-0"
        grade = "Benign"
    else:
        folder_path = r"E:\liuzhou_breastcancer\figure-res-1"
        grade = "Malignant"

    img_name = grade + str(datalist.loc[datalist0, "patient_name"])

    mask_name = datalist.iloc[datalist0, 2]
    if mask_name.endswith(".nii.gz"):
        mask_name = mask_name[:-7]
    elif mask_name.endswith(".nii"):
        mask_name = mask_name[:-4]

    print(f"------------------ {img_name} + {mask_name} is processing. ------------------")

    img_path_in = os.path.join(folder_path, "washIn", img_name + ".nii")
    img_path_out = os.path.join(folder_path, "washOut", img_name + ".nii")
    mask_path = datalist.iloc[datalist0, 7]

    if not os.path.exists(img_path_in):
        print(f"Missing file: {img_path_in}")
        continue

    if not os.path.exists(img_path_out):
        print(f"Missing file: {img_path_out}")
        continue

    curve_done = process_case(
        fig_path=fig_path,
        img_name=img_name,
        mask_name=mask_name,
        image_in_path=img_path_in,
        image_out_path=img_path_out,
        mask_path=mask_path,
        smooth_sigma=1.0,
    )

    if not curve_done:
        print(f"Failed to draw curves for {img_name}")
        continue

    print(f"{img_name} - {mask_name} curve figure is saved.")
