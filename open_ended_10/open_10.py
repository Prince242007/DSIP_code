
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import time

# ==================================================
# OPEN-ENDED 10: FREQUENCY-DOMAIN IMAGE PROCESSING
# ==================================================

INPUT_FOLDER = "open_ended_7"
OUTPUT_FOLDER = "open_ended_10_results"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

IMAGE_EXTENSIONS = [
    "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"
]

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
# 1. Frequency-domain Gaussian low-pass filter
# --------------------------------------------------

def gaussian_low_pass(image, cutoff=40):
    """
    Smooth the image in the frequency domain.
    A Gaussian low-pass filter attenuates high frequencies.
    """

    image_float = image.astype(np.float32)

    # Fourier transform
    spectrum = np.fft.fft2(image_float)
    spectrum_shifted = np.fft.fftshift(spectrum)

    height, width = image.shape

    y, x = np.indices((height, width))
    center_y = height // 2
    center_x = width // 2

    distance = np.sqrt(
        (x - center_x) ** 2 + (y - center_y) ** 2
    )

    # Gaussian low-pass transfer function
    transfer = np.exp(
        -(distance ** 2) / (2 * cutoff ** 2)
    )

    filtered_spectrum = spectrum_shifted * transfer

    # Inverse Fourier transform
    inverse_shifted = np.fft.ifftshift(filtered_spectrum)
    result = np.fft.ifft2(inverse_shifted)

    result = np.real(result)

    return np.clip(result, 0, 255).astype(np.uint8)


# --------------------------------------------------
# 2. Frequency-domain Butterworth low-pass filter
# --------------------------------------------------

def butterworth_low_pass(image, cutoff=40, order=2):
    """
    Butterworth low-pass filter.
    Order controls the steepness of the frequency response.
    """

    image_float = image.astype(np.float32)

    spectrum = np.fft.fft2(image_float)
    spectrum_shifted = np.fft.fftshift(spectrum)

    height, width = image.shape

    y, x = np.indices((height, width))
    center_y = height // 2
    center_x = width // 2

    distance = np.sqrt(
        (x - center_x) ** 2 + (y - center_y) ** 2
    )

    transfer = 1.0 / (
        1.0 + (distance / cutoff) ** (2 * order)
    )

    filtered_spectrum = spectrum_shifted * transfer

    result = np.fft.ifft2(
        np.fft.ifftshift(filtered_spectrum)
    )

    result = np.real(result)

    return np.clip(result, 0, 255).astype(np.uint8)


# --------------------------------------------------
# 3. Frequency-domain high-boost sharpening
# --------------------------------------------------

def high_boost_sharpen(image, cutoff=40, amount=1.5):
    """
    High-boost filtering:
    H_highboost = 1 + amount * (1 - H_lowpass)

    The low-pass component is subtracted from the original
    frequency spectrum to emphasize fine details.
    """

    image_float = image.astype(np.float32)

    spectrum = np.fft.fft2(image_float)
    spectrum_shifted = np.fft.fftshift(spectrum)

    height, width = image.shape

    y, x = np.indices((height, width))
    center_y = height // 2
    center_x = width // 2

    distance = np.sqrt(
        (x - center_x) ** 2 + (y - center_y) ** 2
    )

    low_pass = np.exp(
        -(distance ** 2) / (2 * cutoff ** 2)
    )

    # High-boost transfer function
    high_boost = 1.0 + amount * (1.0 - low_pass)

    enhanced_spectrum = spectrum_shifted * high_boost

    result = np.fft.ifft2(
        np.fft.ifftshift(enhanced_spectrum)
    )

    result = np.real(result)

    return np.clip(result, 0, 255).astype(np.uint8)


# --------------------------------------------------
# 4. Automatic Canny boundary extraction
# --------------------------------------------------

def extract_boundaries(image):
    """
    Generate a binary boundary image using automatically
    calculated Canny thresholds.
    """

    median_intensity = np.median(image)

    lower = int(max(0, 0.66 * median_intensity))
    upper = int(min(255, 1.33 * median_intensity))

    # Ensure that thresholds are different
    if upper <= lower:
        upper = min(255, lower + 50)

    edges = cv2.Canny(image, lower, upper)

    # Optional morphological closing connects small gaps.
    kernel = np.ones((3, 3), np.uint8)

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Ensure strictly binary output: 0 or 255
    edges = np.where(edges > 0, 255, 0).astype(np.uint8)

    return edges, lower, upper


# --------------------------------------------------
# 5. Quantitative measurements
# --------------------------------------------------

def calculate_metrics(image):
    sharpness = cv2.Laplacian(
        image, cv2.CV_64F
    ).var()

    histogram = cv2.calcHist(
        [image], [0], None, [256], [0, 256]
    ).ravel()

    probabilities = histogram / histogram.sum()
    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(
        probabilities * np.log2(probabilities)
    )

    return sharpness, entropy


def edge_density(edges):
    return 100.0 * np.count_nonzero(edges) / edges.size


# --------------------------------------------------
# 6. Process all images
# --------------------------------------------------

report = [
    "OPEN-ENDED 10: FREQUENCY-DOMAIN PROCESSING",
    "=" * 75,
    "No clean reference image is assumed.",
    "Sharpness, entropy and edge density are descriptive indicators.",
    ""
]

for image_path in image_paths:

    filename = os.path.basename(image_path)
    image_name = os.path.splitext(filename)[0]

    print("\nProcessing:", filename)

    color = cv2.imread(image_path)

    if color is None:
        print("Could not read image:", filename)
        continue

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    image_output = os.path.join(
        OUTPUT_FOLDER, image_name
    )
    os.makedirs(image_output, exist_ok=True)

    cv2.imwrite(
        os.path.join(image_output, "original_gray.png"),
        gray
    )

    # --------------------------------------------------
    # 6.1 Frequency-domain smoothing
    # --------------------------------------------------

    start = time.perf_counter()
    gaussian = gaussian_low_pass(gray, cutoff=40)
    gaussian_time = time.perf_counter() - start

    start = time.perf_counter()
    butterworth = butterworth_low_pass(
        gray, cutoff=40, order=2
    )
    butterworth_time = time.perf_counter() - start

    cv2.imwrite(
        os.path.join(image_output, "gaussian_low_pass.png"),
        gaussian
    )

    cv2.imwrite(
        os.path.join(image_output, "butterworth_low_pass.png"),
        butterworth
    )

    # --------------------------------------------------
    # 6.2 High-boost sharpening
    # --------------------------------------------------

    # Apply sharpening to both smoothed versions
    start = time.perf_counter()
    gaussian_sharp = high_boost_sharpen(
        gaussian, cutoff=40, amount=1.5
    )
    gaussian_sharp_time = time.perf_counter() - start

    start = time.perf_counter()
    butterworth_sharp = high_boost_sharpen(
        butterworth, cutoff=40, amount=1.5
    )
    butterworth_sharp_time = time.perf_counter() - start

    cv2.imwrite(
        os.path.join(image_output, "gaussian_high_boost.png"),
        gaussian_sharp
    )

    cv2.imwrite(
        os.path.join(image_output, "butterworth_high_boost.png"),
        butterworth_sharp
    )

    # --------------------------------------------------
    # 6.3 Boundary maps
    # --------------------------------------------------

    original_edges, original_low, original_high = (
        extract_boundaries(gray)
    )

    gaussian_edges, gaussian_low, gaussian_high = (
        extract_boundaries(gaussian_sharp)
    )

    butterworth_edges, butterworth_low, butterworth_high = (
        extract_boundaries(butterworth_sharp)
    )

    cv2.imwrite(
        os.path.join(image_output, "original_boundaries.png"),
        original_edges
    )

    cv2.imwrite(
        os.path.join(image_output, "gaussian_boundaries.png"),
        gaussian_edges
    )

    cv2.imwrite(
        os.path.join(image_output, "butterworth_boundaries.png"),
        butterworth_edges
    )

    # --------------------------------------------------
    # 6.4 Display all processing stages
    # --------------------------------------------------

    display_images = [
        ("Original", gray),
        ("Gaussian Low-Pass", gaussian),
        ("Butterworth Low-Pass", butterworth),
        ("Gaussian High-Boost", gaussian_sharp),
        ("Butterworth High-Boost", butterworth_sharp),
        ("Original Boundaries", original_edges),
        ("Gaussian Boundaries", gaussian_edges),
        ("Butterworth Boundaries", butterworth_edges)
    ]

    plt.figure(figsize=(16, 10))

    for index, (title, output_image) in enumerate(
        display_images
    ):
        plt.subplot(2, 4, index + 1)
        plt.imshow(
            output_image,
            cmap="gray",
            vmin=0,
            vmax=255
        )
        plt.title(title)
        plt.axis("off")

    plt.suptitle(
        f"Frequency-Domain Filtering: {filename}"
    )
    plt.tight_layout()

    plt.savefig(
        os.path.join(image_output, "comparison.png"),
        dpi=150
    )

    plt.show()
    plt.close()

    # --------------------------------------------------
    # 6.5 Quantitative analysis
    # --------------------------------------------------

    report.append(f"IMAGE: {filename}")
    report.append("-" * 75)
    report.append(
        f"{'Method':<27} {'Sharpness':>12} "
        f"{'Entropy':>10} {'Edge Density %':>16}"
    )

    metric_images = [
        ("Original", gray),
        ("Gaussian Low-Pass", gaussian),
        ("Butterworth Low-Pass", butterworth),
        ("Gaussian High-Boost", gaussian_sharp),
        ("Butterworth High-Boost", butterworth_sharp)
    ]

    for name, processed in metric_images:

        sharpness, entropy = calculate_metrics(processed)

        if name == "Original":
            boundary_image = original_edges
        elif name.startswith("Gaussian"):
            boundary_image = gaussian_edges
        else:
            boundary_image = butterworth_edges

        report.append(
            f"{name:<27} {sharpness:>12.2f} "
            f"{entropy:>10.3f} "
            f"{edge_density(boundary_image):>15.2f}"
        )

    report.append(
        f"Gaussian smoothing time: {gaussian_time:.4f} seconds"
    )
    report.append(
        f"Butterworth smoothing time: "
        f"{butterworth_time:.4f} seconds"
    )
    report.append(
        f"Gaussian high-boost time: "
        f"{gaussian_sharp_time:.4f} seconds"
    )
    report.append(
        f"Butterworth high-boost time: "
        f"{butterworth_sharp_time:.4f} seconds"
    )

    report.append(
        "Original Canny thresholds: "
        f"{original_low}, {original_high}"
    )
    report.append(
        "Gaussian boundary Canny thresholds: "
        f"{gaussian_low}, {gaussian_high}"
    )
    report.append(
        "Butterworth boundary Canny thresholds: "
        f"{butterworth_low}, {butterworth_high}"
    )

    report.append("")

    print("Saved processed images to:", image_output)


# --------------------------------------------------
# 7. Save report
# --------------------------------------------------

report_path = os.path.join(
    OUTPUT_FOLDER, "open_ended_10_report.txt"
)

with open(report_path, "w", encoding="utf-8") as file:
    file.write("\n".join(report))

print("\nFrequency-domain experiment completed.")
print("Processed images:", OUTPUT_FOLDER)
print("Report:", report_path)
