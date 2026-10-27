
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import time

# ==================================================
# OPEN-ENDED 5: NON-LINEAR IMAGE FILTERING
# ==================================================

INPUT_FOLDER = "open_ended_7"
OUTPUT_FOLDER = "open_ended_5_results"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]

image_paths = []
for extension in IMAGE_EXTENSIONS:
    image_paths.extend(
        glob.glob(os.path.join(INPUT_FOLDER, extension))
    )
    image_paths.extend(
        glob.glob(os.path.join(INPUT_FOLDER, extension.upper()))
    )

image_paths = sorted(set(image_paths))

if not image_paths:
    print("Error: No images found in", INPUT_FOLDER)
    raise SystemExit

print("Images found:", len(image_paths))


# --------------------------------------------------
# 1. Alpha-trimmed mean filter
# --------------------------------------------------

def alpha_trimmed_filter(image, kernel_size=3, trim_count=2):
    """
    Sort neighborhood values, remove the lowest and highest
    values, and average the remaining values.
    """

    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    if trim_count >= kernel_size * kernel_size:
        raise ValueError("Trim count is too large.")

    padded = cv2.copyMakeBorder(
        image,
        kernel_size // 2,
        kernel_size // 2,
        kernel_size // 2,
        kernel_size // 2,
        cv2.BORDER_REFLECT
    )

    height, width = image.shape
    result = np.zeros_like(image, dtype=np.uint8)

    # Process one row at a time to reduce memory usage.
    for row in range(height):
        row_data = padded[
            row:row + kernel_size,
            :
        ]

        windows = np.lib.stride_tricks.sliding_window_view(
            row_data, (kernel_size, kernel_size)
        )

        windows = windows[0].reshape(width, -1)
        sorted_values = np.sort(windows, axis=1)

        if trim_count > 0:
            sorted_values = sorted_values[
                :, trim_count:-trim_count
            ]

        result[row] = np.clip(
            np.mean(sorted_values, axis=1),
            0, 255
        ).astype(np.uint8)

    return result


# --------------------------------------------------
# 2. Calculate image quality indicators
# --------------------------------------------------

def calculate_metrics(image):
    # Sharpness indicator
    sharpness = cv2.Laplacian(
        image, cv2.CV_64F
    ).var()

    # Entropy: measure of intensity distribution
    histogram = cv2.calcHist(
        [image], [0], None, [256], [0, 256]
    ).ravel()

    probabilities = histogram / histogram.sum()
    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    # Percentage of pixels detected as edges
    edges = cv2.Canny(image, 100, 200)
    edge_density = 100.0 * np.count_nonzero(edges) / edges.size

    return sharpness, entropy, edge_density


# --------------------------------------------------
# 3. Process every image
# --------------------------------------------------

report_lines = [
    "OPEN-ENDED 5: NON-LINEAR IMAGE FILTERING",
    "=" * 65,
    "Metrics are indicators, not absolute quality scores.",
    "No clean reference image is assumed.",
    ""
]

for image_path in image_paths:

    filename = os.path.basename(image_path)
    image_name = os.path.splitext(filename)[0]

    print("\nProcessing:", filename)

    color = cv2.imread(image_path)

    if color is None:
        print("Could not read:", filename)
        continue

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    # Create an output folder for this image
    image_output = os.path.join(OUTPUT_FOLDER, image_name)
    os.makedirs(image_output, exist_ok=True)

    # Dictionary containing all filter results
    results = {
        "Original": gray
    }

    # --------------------------------------------------
    # 3.1 Median filters
    # --------------------------------------------------

    for size in [3, 5, 7]:
        start = time.perf_counter()

        filtered = cv2.medianBlur(gray, size)

        elapsed = time.perf_counter() - start

        results[f"Median {size}x{size}"] = filtered

        cv2.imwrite(
            os.path.join(
                image_output, f"median_{size}x{size}.png"
            ),
            filtered
        )

        print(f"Median {size}x{size}: {elapsed:.4f} seconds")

    # --------------------------------------------------
    # 3.2 Bilateral filter
    # --------------------------------------------------

    start = time.perf_counter()

    bilateral = cv2.bilateralFilter(
        gray,
        d=9,
        sigmaColor=50,
        sigmaSpace=50
    )

    bilateral_time = time.perf_counter() - start

    results["Bilateral"] = bilateral

    cv2.imwrite(
        os.path.join(image_output, "bilateral.png"),
        bilateral
    )

    # --------------------------------------------------
    # 3.3 Alpha-trimmed mean filter
    # --------------------------------------------------

    start = time.perf_counter()

    alpha_trimmed = alpha_trimmed_filter(
        gray,
        kernel_size=3,
        trim_count=2
    )

    alpha_time = time.perf_counter() - start

    results["Alpha-trimmed"] = alpha_trimmed

    cv2.imwrite(
        os.path.join(image_output, "alpha_trimmed.png"),
        alpha_trimmed
    )

    # --------------------------------------------------
    # 3.4 Morphological filtering
    # --------------------------------------------------

    kernel = np.ones((3, 3), np.uint8)

    # Opening suppresses small bright regions.
    opened = cv2.morphologyEx(
        gray, cv2.MORPH_OPEN, kernel
    )

    # Closing suppresses small dark regions.
    closed = cv2.morphologyEx(
        gray, cv2.MORPH_CLOSE, kernel
    )

    # Average the two results as a simple combined experiment.
    morphological = cv2.addWeighted(
        opened, 0.5, closed, 0.5, 0
    )

    results["Morphological"] = morphological

    cv2.imwrite(
        os.path.join(image_output, "morphological.png"),
        morphological
    )

    # --------------------------------------------------
    # 4. Save original and filtered images
    # --------------------------------------------------

    cv2.imwrite(
        os.path.join(image_output, "original_gray.png"),
        gray
    )

    # Comparison figure
    names = list(results.keys())

    plt.figure(figsize=(15, 9))

    for index, name in enumerate(names):
        plt.subplot(2, 4, index + 1)
        plt.imshow(results[name], cmap="gray", vmin=0, vmax=255)
        plt.title(name)
        plt.axis("off")

    plt.suptitle(
        f"Non-Linear Filter Comparison: {filename}"
    )
    plt.tight_layout()

    comparison_path = os.path.join(
        image_output, "filter_comparison.png"
    )

    plt.savefig(comparison_path, dpi=150)
    plt.show()
    plt.close()

    # --------------------------------------------------
    # 5. Quantitative comparison
    # --------------------------------------------------

    report_lines.append(f"IMAGE: {filename}")
    report_lines.append("-" * 65)
    report_lines.append(
        f"{'Method':<22} {'Sharpness':>12} "
        f"{'Entropy':>10} {'Edge %':>10}"
    )

    for name, filtered in results.items():
        sharpness, entropy, edge_density = calculate_metrics(
            filtered
        )

        report_lines.append(
            f"{name:<22} {sharpness:>12.2f} "
            f"{entropy:>10.3f} {edge_density:>9.2f}"
        )

    report_lines.append(
        "Bilateral execution time: "
        f"{bilateral_time:.4f} seconds"
    )
    report_lines.append(
        "Alpha-trimmed execution time: "
        f"{alpha_time:.4f} seconds"
    )

    report_lines.append("")

    print("Saved results to:", image_output)


# --------------------------------------------------
# 6. Save the experiment report
# --------------------------------------------------

report_path = os.path.join(
    OUTPUT_FOLDER, "open_ended_5_report.txt"
)

with open(report_path, "w", encoding="utf-8") as file:
    file.write("\n".join(report_lines))

print("\nExperiment completed.")
print("Processed images:", OUTPUT_FOLDER)
print("Report:", report_path)
