import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Dental ImageJ Processor",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    h1 {
        margin-bottom: 0.15rem;
    }

    .subtitle {
        color: #777;
        margin-bottom: 1.5rem;
    }

    .mode-box {
        padding: 0.7rem 0.9rem;
        border-radius: 10px;
        background: rgba(120,120,120,0.08);
        margin-bottom: 0.8rem;
    }

    .small-note {
        color: #777;
        font-size: 0.85rem;
    }

    div[data-testid="stImage"] img {
        border-radius: 6px;
    }

    .stDownloadButton button {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TITLE
# ============================================================

st.title("🦷 Dental ImageJ Processor")

st.markdown(
    '<div class="subtitle">'
    "Dental radiograph enhancement using ImageJ/Fiji-style image-processing operations"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_to_uint8(image):
    """
    Convert arbitrary numeric image data to uint8.

    Percentile normalization is used for floating-point and
    higher-bit-depth images so that display is robust.
    """
    image = np.asarray(image)

    if image.dtype == np.uint8:
        return image.copy()

    image = image.astype(np.float32)

    finite = np.isfinite(image)

    if not np.any(finite):
        return np.zeros(image.shape, dtype=np.uint8)

    valid = image[finite]

    low = np.percentile(valid, 0.5)
    high = np.percentile(valid, 99.5)

    if high <= low:
        low = float(valid.min())
        high = float(valid.max())

    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)

    image = (image - low) / (high - low)
    image = np.clip(image, 0, 1)

    return (image * 255).astype(np.uint8)


def ensure_gray(image):
    """
    Convert an image to grayscale.
    """
    image = np.asarray(image)

    if image.ndim == 2:
        return image

    if image.ndim == 3:
        if image.shape[2] == 1:
            return image[:, :, 0]

        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    raise ValueError("Unsupported image dimensions.")


def prepare_image(image):
    """
    Convert input to display/processable uint8 grayscale.
    """
    image = ensure_gray(image)
    return normalize_to_uint8(image)


def resize_for_display(image, max_width=1400, max_height=1000):
    """
    Resize only for display.
    Processing is performed on the original resolution.
    """
    h, w = image.shape[:2]

    scale = min(
        max_width / max(w, 1),
        max_height / max(h, 1),
        1.0,
    )

    if scale >= 1:
        return image

    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )


def image_to_png_bytes(image):
    """
    Convert grayscale uint8 image to PNG bytes.
    """
    image = normalize_to_uint8(image)

    ok, encoded = cv2.imencode(".png", image)

    if not ok:
        raise ValueError("Could not encode image.")

    return encoded.tobytes()


# ============================================================
# DICOM
# ============================================================

def load_dicom(file_bytes):
    """
    Load DICOM pixel data and apply common modality transformations.
    """
    if not DICOM_AVAILABLE:
        raise RuntimeError(
            "pydicom is not installed. Add pydicom to requirements.txt."
        )

    ds = pydicom.dcmread(io.BytesIO(file_bytes))

    pixels = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    pixels = pixels * slope + intercept

    photometric = str(
        getattr(ds, "PhotometricInterpretation", "")
    ).upper()

    if photometric == "MONOCHROME1":
        pixels = np.max(pixels) - pixels

    # Apply window center/width when available.
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)

    if wc is not None and ww is not None:
        try:
            if isinstance(wc, pydicom.multival.MultiValue):
                wc = float(wc[0])
            else:
                wc = float(wc)

            if isinstance(ww, pydicom.multival.MultiValue):
                ww = float(ww[0])
            else:
                ww = float(ww)

            if ww > 0:
                low = wc - ww / 2.0
                high = wc + ww / 2.0
                pixels = np.clip(pixels, low, high)

        except Exception:
            pass

    return normalize_to_uint8(pixels)


def load_uploaded_image(uploaded_file):
    """
    Load PNG/JPG/TIFF/DICOM.
    """
    data = uploaded_file.getvalue()

    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix in [".dcm", ".dicom"]:
        return load_dicom(data)

    try:
        pil = Image.open(io.BytesIO(data))
        arr = np.array(pil)
        return prepare_image(arr)

    except Exception:
        # Some DICOM files do not have .dcm extension.
        if DICOM_AVAILABLE:
            try:
                return load_dicom(data)
            except Exception:
                pass

        raise ValueError(
            "Could not read this file as an image or DICOM."
        )


# ============================================================
# IMAGEJ-STYLE CONTRAST
# ============================================================

def imagej_enhance_contrast(
    image,
    saturated=0.35,
    normalize=True,
):
    """
    ImageJ-style histogram stretching.

    ImageJ's Enhance Contrast command uses histogram stretching
    and optionally normalization/equalization.
    """
    image = image.astype(np.float32)

    flat = image.ravel()

    low = np.percentile(flat, saturated / 2.0)
    high = np.percentile(flat, 100.0 - saturated / 2.0)

    if high <= low:
        return normalize_to_uint8(image)

    result = (image - low) * 255.0 / (high - low)

    result = np.clip(result, 0, 255)

    if normalize:
        return result.astype(np.uint8)

    return result.astype(np.uint8)


def histogram_equalization(image):
    return cv2.equalizeHist(
        normalize_to_uint8(image)
    )


# ============================================================
# CLAHE
# ============================================================

def imagej_clahe(
    image,
    block_size=127,
    bins=256,
    max_slope=3.0,
):
    """
    Practical OpenCV implementation of CLAHE.

    ImageJ/Fiji's CLAHE exposes:
        block size
        histogram bins
        maximum slope

    OpenCV uses:
        tile grid
        clip limit

    The mapping below provides a controllable ImageJ-like
    interface while using OpenCV's optimized CLAHE engine.
    """
    image = normalize_to_uint8(image)

    h, w = image.shape

    tile_x = max(2, int(round(w / block_size)))
    tile_y = max(2, int(round(h / block_size)))

    tile_x = min(tile_x, 32)
    tile_y = min(tile_y, 32)

    # Map ImageJ-like slope to OpenCV clip limit.
    clip_limit = max(1.0, float(max_slope))

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_x, tile_y),
    )

    return clahe.apply(image)


# ============================================================
# GAUSSIAN
# ============================================================

def gaussian_blur(image, sigma):
    image = normalize_to_uint8(image)

    sigma = float(sigma)

    if sigma <= 0:
        return image.copy()

    k = max(3, int(round(sigma * 6)) | 1)

    return cv2.GaussianBlur(
        image,
        (k, k),
        sigmaX=sigma,
        sigmaY=sigma,
    )


# ============================================================
# MEDIAN
# ============================================================

def median_filter(image, kernel):
    image = normalize_to_uint8(image)

    kernel = int(kernel)

    if kernel < 3:
        kernel = 3

    if kernel % 2 == 0:
        kernel += 1

    return cv2.medianBlur(
        image,
        kernel,
    )


# ============================================================
# MEAN
# ============================================================

def mean_filter(image, radius):
    image = normalize_to_uint8(image)

    radius = int(radius)

    if radius < 1:
        return image.copy()

    size = radius * 2 + 1

    return cv2.blur(
        image,
        (size, size),
    )


# ============================================================
# IMAGEJ UNSHARP MASK
# ============================================================

def imagej_unsharp_mask(
    image,
    sigma=1.0,
    weight=0.6,
):
    """
    ImageJ-style Unsharp Mask.

    Conceptually:

        blurred = Gaussian(image)
        highpass = image - blurred
        result = image + weight * highpass

    ImageJ describes Radius (Sigma) as the Gaussian sigma
    and Mask Weight as the strength of the high-pass component.
    """
    image = normalize_to_uint8(image)

    original = image.astype(np.float32)

    blurred = cv2.GaussianBlur(
        original,
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
    )

    highpass = original - blurred

    result = original + float(weight) * highpass

    result = np.clip(result, 0, 255)

    return result.astype(np.uint8)


# ============================================================
# IMAGEJ SHARPEN
# ============================================================

def imagej_sharpen(image, strength=1.0):
    """
    ImageJ-style Sharpen operation.

    ImageJ documents a 3x3 sharpening convolution:

        -1 -1 -1
        -1 12 -1
        -1 -1 -1

    The kernel is normalized here to produce a stable uint8 result.
    """
    image = normalize_to_uint8(image)

    strength = float(strength)

    base = np.array(
        [
            [-1, -1, -1],
            [-1, 12, -1],
            [-1, -1, -1],
        ],
        dtype=np.float32,
    )

    kernel = np.eye(3, dtype=np.float32)

    kernel += (base - kernel) * strength

    result = cv2.filter2D(
        image.astype(np.float32),
        -1,
        kernel,
        borderType=cv2.BORDER_REPLICATE,
    )

    return normalize_to_uint8(result)


# ============================================================
# SOBEL / FIND EDGES
# ============================================================

def sobel_edges(image, strength=1.0):
    """
    ImageJ-style Find Edges based on Sobel derivatives.
    """
    image = normalize_to_uint8(image)

    img = image.astype(np.float32)

    gx = cv2.Sobel(
        img,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        img,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = cv2.magnitude(gx, gy)

    magnitude *= float(strength)

    return normalize_to_uint8(magnitude)


# ============================================================
# LAPLACIAN
# ============================================================

def laplacian_enhance(
    image,
    strength=0.5,
):
    """
    Laplacian-based detail enhancement.
    """
    image = normalize_to_uint8(image)

    img = image.astype(np.float32)

    lap = cv2.Laplacian(
        img,
        cv2.CV_32F,
        ksize=3,
    )

    result = img - float(strength) * lap

    return normalize_to_uint8(result)


# ============================================================
# CONVOLUTION
# ============================================================

CONVOLUTION_KERNELS = {
    "Identity": np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.float32,
    ),

    "Sharpen": np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    ),

    "Strong Sharpen": np.array(
        [
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1],
        ],
        dtype=np.float32,
    ),

    "Edge Detection": np.array(
        [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1],
        ],
        dtype=np.float32,
    ),

    "Sobel X": np.array(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ],
        dtype=np.float32,
    ),

    "Sobel Y": np.array(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ],
        dtype=np.float32,
    ),

    "Laplacian": np.array(
        [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0],
        ],
        dtype=np.float32,
    ),

    "Emboss": np.array(
        [
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2],
        ],
        dtype=np.float32,
    ),
}


def convolve_image(image, kernel_name, strength=1.0):
    image = normalize_to_uint8(image)

    kernel = CONVOLUTION_KERNELS[kernel_name].copy()

    identity = np.zeros((3, 3), dtype=np.float32)
    identity[1, 1] = 1.0

    kernel = identity + float(strength) * (kernel - identity)

    result = cv2.filter2D(
        image.astype(np.float32),
        -1,
        kernel,
        borderType=cv2.BORDER_REPLICATE,
    )

    return normalize_to_uint8(result)


# ============================================================
# MINIMUM / MAXIMUM
# ============================================================

def minimum_filter(image, radius):
    image = normalize_to_uint8(image)

    radius = max(1, int(radius))

    size = radius * 2 + 1

    kernel = np.ones(
        (size, size),
        dtype=np.uint8,
    )

    return cv2.erode(
        image,
        kernel,
        borderType=cv2.BORDER_REPLICATE,
    )


def maximum_filter(image, radius):
    image = normalize_to_uint8(image)

    radius = max(1, int(radius))

    size = radius * 2 + 1

    kernel = np.ones(
        (size, size),
        dtype=np.uint8,
    )

    return cv2.dilate(
        image,
        kernel,
        borderType=cv2.BORDER_REPLICATE,
    )


# ============================================================
# MORPHOLOGY
# ============================================================

def morphological_open(image, radius):
    image = normalize_to_uint8(image)

    radius = max(1, int(radius))

    size = radius * 2 + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (size, size),
    )

    return cv2.morphologyEx(
        image,
        cv2.MORPH_OPEN,
        kernel,
    )


def morphological_close(image, radius):
    image = normalize_to_uint8(image)

    radius = max(1, int(radius))

    size = radius * 2 + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (size, size),
    )

    return cv2.morphologyEx(
        image,
        cv2.MORPH_CLOSE,
        kernel,
    )


# ============================================================
# BACKGROUND SUBTRACTION
# ============================================================

def subtract_background(
    image,
    radius=25,
):
    """
    Rolling-ball-like background correction using morphological
    opening. It is designed to provide the same type of operation
    as ImageJ's Subtract Background command.
    """
    image = normalize_to_uint8(image)

    radius = max(3, int(radius))

    if radius % 2 == 0:
        radius += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius, radius),
    )

    background = cv2.morphologyEx(
        image,
        cv2.MORPH_OPEN,
        kernel,
    )

    result = (
        image.astype(np.float32)
        - background.astype(np.float32)
        + 128.0
    )

    return normalize_to_uint8(result)


# ============================================================
# GAMMA
# ============================================================

def gamma_correction(image, gamma):
    image = normalize_to_uint8(image)

    gamma = float(gamma)

    if gamma <= 0:
        gamma = 1.0

    normalized = image.astype(np.float32) / 255.0

    result = np.power(
        normalized,
        gamma,
    )

    return np.clip(
        result * 255.0,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# DENTAL PRESETS
# ============================================================

def dental_endo(image):
    """
    Endodontic preset.

    CLAHE
        ->
    mild Gaussian smoothing
        ->
    ImageJ-style Unsharp Mask
    """
    result = imagej_clahe(
        image,
        block_size=80,
        bins=256,
        max_slope=2.5,
    )

    result = gaussian_blur(
        result,
        sigma=0.45,
    )

    result = imagej_unsharp_mask(
        result,
        sigma=0.9,
        weight=1.8,
    )

    return result


def dental_perio(image):
    """
    Periodontal / bone preset.

    Contrast enhancement
        ->
    CLAHE
        ->
    moderate Unsharp Mask
    """
    result = imagej_enhance_contrast(
        image,
        saturated=0.35,
        normalize=True,
    )

    result = imagej_clahe(
        result,
        block_size=100,
        bins=256,
        max_slope=2.2,
    )

    result = imagej_unsharp_mask(
        result,
        sigma=1.5,
        weight=1.1,
    )

    return result


def dental_imagej_preset(image):
    """
    General-purpose ImageJ-style dental enhancement.
    """
    result = imagej_enhance_contrast(
        image,
        saturated=0.35,
        normalize=True,
    )

    result = imagej_clahe(
        result,
        block_size=100,
        bins=256,
        max_slope=2.0,
    )

    result = imagej_unsharp_mask(
        result,
        sigma=1.0,
        weight=0.8,
    )

    return result


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Image")

    uploaded = st.file_uploader(
        "Upload dental image",
        type=[
            "png",
            "jpg",
            "jpeg",
            "tif",
            "tiff",
            "dcm",
            "dicom",
        ],
        help="PNG, JPG, TIFF and DICOM are supported.",
    )

    st.divider()

    st.header("Processing mode")

    mode = st.selectbox(
        "Choose operation",
        [
            "ImageJ Dental Preset",
            "Endodontic Preset",
            "Perio / Bone Preset",
            "Enhance Contrast",
            "Histogram Equalization",
            "CLAHE",
            "Gaussian Blur",
            "Median Filter",
            "Mean Filter",
            "Unsharp Mask",
            "ImageJ Sharpen",
            "Find Edges / Sobel",
            "Laplacian Enhancement",
            "Convolve",
            "Minimum",
            "Maximum",
            "Morphological Opening",
            "Morphological Closing",
            "Subtract Background",
            "Gamma Correction",
        ],
    )

    st.divider()


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded is None:

    st.info(
        "Upload a dental radiograph from the sidebar to begin."
    )

    st.markdown(
        """
        ### Supported processing

        **Contrast**
        - Enhance Contrast
        - Histogram Equalization
        - CLAHE

        **Noise reduction**
        - Gaussian Blur
        - Median
        - Mean

        **Sharpening**
        - ImageJ Unsharp Mask
        - ImageJ Sharpen
        - Laplacian enhancement

        **Edges / convolution**
        - Sobel / Find Edges
        - Custom convolution kernels

        **Morphology**
        - Minimum
        - Maximum
        - Opening
        - Closing

        **Background**
        - Rolling-ball-style background subtraction

        **Dental presets**
        - General ImageJ Dental
        - Endodontic
        - Perio / Bone
        """
    )

    st.stop()


# ============================================================
# LOAD IMAGE
# ============================================================

try:
    original = load_uploaded_image(uploaded)

except Exception as exc:

    st.error(
        f"Could not load the image: {exc}"
    )

    st.stop()


original = normalize_to_uint8(original)


# ============================================================
# MODE-SPECIFIC CONTROLS
# ============================================================

with st.sidebar:

    if mode == "ImageJ Dental Preset":

        st.subheader("Dental preset")

        st.caption(
            "General-purpose ImageJ-style dental enhancement."
        )

        preset_strength = st.slider(
            "Final sharpening",
            0.0,
            2.0,
            0.8,
            0.05,
        )


    elif mode == "Endodontic Preset":

        st.subheader("Endodontic")

        clahe_slope = st.slider(
            "CLAHE maximum slope",
            1.0,
            5.0,
            2.5,
            0.1,
        )

        usm_sigma = st.slider(
            "Unsharp Sigma",
            0.3,
            2.0,
            0.9,
            0.05,
        )

        usm_weight = st.slider(
            "Unsharp weight",
            0.0,
            3.0,
            1.8,
            0.05,
        )


    elif mode == "Perio / Bone Preset":

        st.subheader("Perio / Bone")

        contrast = st.slider(
            "Saturated pixels (%)",
            0.05,
            2.0,
            0.35,
            0.05,
        )

        usm_sigma = st.slider(
            "Unsharp Sigma",
            0.5,
            3.0,
            1.5,
            0.05,
        )

        usm_weight = st.slider(
            "Unsharp weight",
            0.0,
            2.5,
            1.1,
            0.05,
        )


    elif mode == "Enhance Contrast":

        st.subheader("ImageJ Enhance Contrast")

        saturated = st.slider(
            "Saturated pixels (%)",
            0.01,
            5.0,
            0.35,
            0.01,
        )

        normalize = st.checkbox(
            "Normalize",
            value=True,
        )


    elif mode == "Histogram Equalization":

        st.subheader("Histogram")

        st.caption(
            "Global histogram equalization."
        )


    elif mode == "CLAHE":

        st.subheader("ImageJ-style CLAHE")

        block_size = st.slider(
            "Block size",
            16,
            256,
            100,
            1,
        )

        bins = st.select_slider(
            "Histogram bins",
            options=[64, 128, 256],
            value=256,
        )

        max_slope = st.slider(
            "Maximum slope",
            1.0,
            8.0,
            2.5,
            0.1,
        )


    elif mode == "Gaussian Blur":

        st.subheader("Gaussian Blur")

        sigma = st.slider(
            "Sigma",
            0.1,
            10.0,
            1.0,
            0.1,
        )


    elif mode == "Median Filter":

        st.subheader("Median Filter")

        kernel = st.select_slider(
            "Kernel size",
            options=[3, 5, 7, 9, 11],
            value=3,
        )


    elif mode == "Mean Filter":

        st.subheader("Mean Filter")

        radius = st.slider(
            "Radius",
            1,
            10,
            1,
        )


    elif mode == "Unsharp Mask":

        st.subheader("ImageJ Unsharp Mask")

        sigma = st.slider(
            "Radius / Sigma",
            0.1,
            5.0,
            1.0,
            0.05,
        )

        weight = st.slider(
            "Mask weight",
            0.0,
            5.0,
            0.6,
            0.05,
        )


    elif mode == "ImageJ Sharpen":

        st.subheader("ImageJ Sharpen")

        strength = st.slider(
            "Sharpen strength",
            0.0,
            2.0,
            1.0,
            0.05,
        )


    elif mode == "Find Edges / Sobel":

        st.subheader("Sobel")

        strength = st.slider(
            "Edge strength",
            0.1,
            3.0,
            1.0,
            0.05,
        )


    elif mode == "Laplacian Enhancement":

        st.subheader("Laplacian")

        strength = st.slider(
            "Strength",
            0.0,
            2.0,
            0.5,
            0.05,
        )


    elif mode == "Convolve":

        st.subheader("Convolution")

        kernel_name = st.selectbox(
            "Kernel",
            list(CONVOLUTION_KERNELS.keys()),
        )

        strength = st.slider(
            "Kernel strength",
            0.0,
            2.0,
            1.0,
            0.05,
        )


    elif mode == "Minimum":

        st.subheader("Minimum")

        radius = st.slider(
            "Radius",
            1,
            10,
            1,
        )


    elif mode == "Maximum":

        st.subheader("Maximum")

        radius = st.slider(
            "Radius",
            1,
            10,
            1,
        )


    elif mode == "Morphological Opening":

        st.subheader("Opening")

        radius = st.slider(
            "Radius",
            1,
            10,
            2,
        )


    elif mode == "Morphological Closing":

        st.subheader("Closing")

        radius = st.slider(
            "Radius",
            1,
            10,
            2,
        )


    elif mode == "Subtract Background":

        st.subheader("Background subtraction")

        background_radius = st.slider(
            "Background radius",
            5,
            150,
            25,
            1,
        )


    elif mode == "Gamma Correction":

        st.subheader("Gamma")

        gamma = st.slider(
            "Gamma",
            0.2,
            3.0,
            1.0,
            0.05,
        )


# ============================================================
# PROCESS IMAGE
# ============================================================

processed = original.copy()


if mode == "ImageJ Dental Preset":

    processed = imagej_enhance_contrast(
        processed,
        saturated=0.35,
        normalize=True,
    )

    processed = imagej_clahe(
        processed,
        block_size=100,
        bins=256,
        max_slope=2.0,
    )

    processed = imagej_unsharp_mask(
        processed,
        sigma=1.0,
        weight=preset_strength,
    )


elif mode == "Endodontic Preset":

    processed = imagej_clahe(
        processed,
        block_size=80,
        bins=256,
        max_slope=clahe_slope,
    )

    processed = gaussian_blur(
        processed,
        sigma=0.45,
    )

    processed = imagej_unsharp_mask(
        processed,
        sigma=usm_sigma,
        weight=usm_weight,
    )


elif mode == "Perio / Bone Preset":

    processed = imagej_enhance_contrast(
        processed,
        saturated=contrast,
        normalize=True,
    )

    processed = imagej_clahe(
        processed,
        block_size=100,
        bins=256,
        max_slope=2.2,
    )

    processed = imagej_unsharp_mask(
        processed,
        sigma=usm_sigma,
        weight=usm_weight,
    )


elif mode == "Enhance Contrast":

    processed = imagej_enhance_contrast(
        processed,
        saturated=saturated,
        normalize=normalize,
    )


elif mode == "Histogram Equalization":

    processed = histogram_equalization(
        processed
    )


elif mode == "CLAHE":

    processed = imagej_clahe(
        processed,
        block_size=block_size,
        bins=bins,
        max_slope=max_slope,
    )


elif mode == "Gaussian Blur":

    processed = gaussian_blur(
        processed,
        sigma=sigma,
    )


elif mode == "Median Filter":

    processed = median_filter(
        processed,
        kernel=kernel,
    )


elif mode == "Mean Filter":

    processed = mean_filter(
        processed,
        radius=radius,
    )


elif mode == "Unsharp Mask":

    processed = imagej_unsharp_mask(
        processed,
        sigma=sigma,
        weight=weight,
    )


elif mode == "ImageJ Sharpen":

    processed = imagej_sharpen(
        processed,
        strength=strength,
    )


elif mode == "Find Edges / Sobel":

    processed = sobel_edges(
        processed,
        strength=strength,
    )


elif mode == "Laplacian Enhancement":

    processed = laplacian_enhance(
        processed,
        strength=strength,
    )


elif mode == "Convolve":

    processed = convolve_image(
        processed,
        kernel_name=kernel_name,
        strength=strength,
    )


elif mode == "Minimum":

    processed = minimum_filter(
        processed,
        radius=radius,
    )


elif mode == "Maximum":

    processed = maximum_filter(
        processed,
        radius=radius,
    )


elif mode == "Morphological Opening":

    processed = morphological_open(
        processed,
        radius=radius,
    )


elif mode == "Morphological Closing":

    processed = morphological_close(
        processed,
        radius=radius,
    )


elif mode == "Subtract Background":

    processed = subtract_background(
        processed,
        radius=background_radius,
    )


elif mode == "Gamma Correction":

    processed = gamma_correction(
        processed,
        gamma=gamma,
    )


# ============================================================
# DISPLAY
# ============================================================

display_original = resize_for_display(original)
display_processed = resize_for_display(processed)

st.subheader(mode)

col1, col2 = st.columns(2)

with col1:

    st.markdown("### Original")

    st.image(
        display_original,
        use_container_width=True,
        clamp=True,
    )


with col2:

    st.markdown("### Processed")

    st.image(
        display_processed,
        use_container_width=True,
        clamp=True,
    )


# ============================================================
# IMAGE INFORMATION
# ============================================================

st.divider()

info1, info2, info3 = st.columns(3)

with info1:
    st.metric(
        "Width",
        f"{original.shape[1]} px",
    )

with info2:
    st.metric(
        "Height",
        f"{original.shape[0]} px",
    )

with info3:
    st.metric(
        "Operation",
        mode,
    )


# ============================================================
# DOWNLOAD
# ============================================================

st.divider()

st.subheader("Export")

png_bytes = image_to_png_bytes(processed)

base_name = Path(uploaded.name).stem

st.download_button(
    label="⬇️ Download processed PNG",
    data=png_bytes,
    file_name=f"{base_name}_{mode.lower().replace(' ', '_').replace('/', '-')}.png",
    mime="image/png",
)


st.caption(
    "Processing is performed on the original image resolution. "
    "The displayed images may be resized only for browser viewing."
)
