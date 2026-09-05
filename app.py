```python
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
    page_title="Dental Image Processor",
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

    .app-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }

    .app-subtitle {
        font-size: 1.15rem;
        color: #666;
        margin-bottom: 0.1rem;
    }

    .app-author {
        font-size: 0.9rem;
        color: #888;
        margin-bottom: 1.5rem;
    }

    .operation-title {
        font-size: 1.35rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
    }

    .info-box {
        padding: 0.7rem 0.9rem;
        border-radius: 10px;
        background: rgba(120, 120, 120, 0.08);
        min-height: 70px;
    }

    .info-label {
        font-size: 0.82rem;
        color: #777;
        margin-bottom: 0.15rem;
    }

    .info-value {
        font-size: 1.05rem;
        font-weight: 600;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
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
# APP HEADER
# ============================================================

st.markdown(
    '<div class="app-title">🦷 Dental Image Processor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    'Επεξεργασία και βελτίωση οδοντιατρικών ακτινογραφιών'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-author">'
    'created by Tasos Dimitrakopoulos 2026'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_to_uint8(image):
    """
    Convert an image of arbitrary numeric depth to uint8.
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
    Convert RGB/RGBA images to grayscale.
    """
    image = np.asarray(image)

    if image.ndim == 2:
        return image

    if image.ndim == 3:

        if image.shape[2] == 1:
            return image[:, :, 0]

        if image.shape[2] == 4:
            return cv2.cvtColor(
                image,
                cv2.COLOR_RGBA2GRAY,
            )

        return cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY,
        )

    raise ValueError("Unsupported image dimensions.")


def prepare_image(image):
    """
    Prepare an uploaded image for processing.
    """
    image = ensure_gray(image)

    return normalize_to_uint8(image)


def resize_for_display(
    image,
    max_width=1400,
    max_height=1000,
):
    """
    Resize only for browser display.
    Processing remains at original resolution.
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
    Convert processed image to PNG bytes.
    """
    image = normalize_to_uint8(image)

    success, encoded = cv2.imencode(
        ".png",
        image,
    )

    if not success:
        raise ValueError(
            "Could not encode processed image."
        )

    return encoded.tobytes()


# ============================================================
# DICOM
# ============================================================

def load_dicom(file_bytes):
    """
    Load a DICOM image and apply common modality transformations.
    """
    if not DICOM_AVAILABLE:
        raise RuntimeError(
            "pydicom is not installed. "
            "Add pydicom to requirements.txt."
        )

    ds = pydicom.dcmread(
        io.BytesIO(file_bytes)
    )

    pixels = ds.pixel_array.astype(
        np.float32
    )

    slope = float(
        getattr(
            ds,
            "RescaleSlope",
            1.0,
        )
    )

    intercept = float(
        getattr(
            ds,
            "RescaleIntercept",
            0.0,
        )
    )

    pixels = pixels * slope + intercept

    photometric = str(
        getattr(
            ds,
            "PhotometricInterpretation",
            "",
        )
    ).upper()

    if photometric == "MONOCHROME1":
        pixels = np.max(pixels) - pixels

    window_center = getattr(
        ds,
        "WindowCenter",
        None,
    )

    window_width = getattr(
        ds,
        "WindowWidth",
        None,
    )

    if (
        window_center is not None
        and window_width is not None
    ):
        try:

            if isinstance(
                window_center,
                pydicom.multival.MultiValue,
            ):
                window_center = float(
                    window_center[0]
                )
            else:
                window_center = float(
                    window_center
                )

            if isinstance(
                window_width,
                pydicom.multival.MultiValue,
            ):
                window_width = float(
                    window_width[0]
                )
            else:
                window_width = float(
                    window_width
                )

            if window_width > 0:

                low = (
                    window_center
                    - window_width / 2.0
                )

                high = (
                    window_center
                    + window_width / 2.0
                )

                pixels = np.clip(
                    pixels,
                    low,
                    high,
                )

        except Exception:
            pass

    return normalize_to_uint8(
        pixels
    )


def load_uploaded_image(
    uploaded_file
):
    """
    Load PNG, JPG, JPEG, TIFF or DICOM.
    """
    data = uploaded_file.getvalue()

    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    if suffix in [
        ".dcm",
        ".dicom",
    ]:
        return load_dicom(data)

    try:

        pil = Image.open(
            io.BytesIO(data)
        )

        array = np.array(pil)

        return prepare_image(array)

    except Exception:

        if DICOM_AVAILABLE:

            try:
                return load_dicom(data)

            except Exception:
                pass

        raise ValueError(
            "Could not read this file as "
            "an image or DICOM."
        )


# ============================================================
# CONTRAST
# ============================================================

def enhance_contrast(
    image,
    saturated=0.35,
):
    """
    Histogram-based contrast stretching.
    """
    image = image.astype(
        np.float32
    )

    flat = image.ravel()

    low = np.percentile(
        flat,
        saturated / 2.0,
    )

    high = np.percentile(
        flat,
        100.0 - saturated / 2.0,
    )

    if high <= low:
        return normalize_to_uint8(
            image
        )

    result = (
        (image - low)
        * 255.0
        / (high - low)
    )

    result = np.clip(
        result,
        0,
        255,
    )

    return result.astype(
        np.uint8
    )


def histogram_equalization(
    image
):
    return cv2.equalizeHist(
        normalize_to_uint8(image)
    )


# ============================================================
# CLAHE
# ============================================================

def apply_clahe(
    image,
    block_size=100,
    bins=256,
    max_slope=2.5,
):
    """
    Local contrast enhancement.
    """
    image = normalize_to_uint8(
        image
    )

    h, w = image.shape

    tile_x = max(
        2,
        int(round(
            w / block_size
        )),
    )

    tile_y = max(
        2,
        int(round(
            h / block_size
        )),
    )

    tile_x = min(
        tile_x,
        32,
    )

    tile_y = min(
        tile_y,
        32,
    )

    clip_limit = max(
        1.0,
        float(max_slope),
    )

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(
            tile_x,
            tile_y,
        ),
    )

    return clahe.apply(
        image
    )


# ============================================================
# GAUSSIAN BLUR
# ============================================================

def gaussian_blur(
    image,
    sigma,
):
    image = normalize_to_uint8(
        image
    )

    sigma = float(sigma)

    if sigma <= 0:
        return image.copy()

    return cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )


# ============================================================
# MEDIAN FILTER
# ============================================================

def median_filter(
    image,
    kernel,
):
    image = normalize_to_uint8(
        image
    )

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
# MEAN FILTER
# ============================================================

def mean_filter(
    image,
    radius,
):
    image = normalize_to_uint8(
        image
    )

    radius = max(
        1,
        int(radius),
    )

    size = radius * 2 + 1

    return cv2.blur(
        image,
        (
            size,
            size,
        ),
    )


# ============================================================
# UNSHARP MASK
# ============================================================

def unsharp_mask(
    image,
    sigma=1.0,
    weight=0.6,
):
    """
    High-frequency detail enhancement.
    """
    image = normalize_to_uint8(
        image
    )

    original = image.astype(
        np.float32
    )

    blurred = cv2.GaussianBlur(
        original,
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
    )

    highpass = (
        original - blurred
    )

    result = (
        original
        + float(weight) * highpass
    )

    result = np.clip(
        result,
        0,
        255,
    )

    return result.astype(
        np.uint8
    )


# ============================================================
# SHARPEN
# ============================================================

def sharpen(
    image,
    strength=1.0,
):
    """
    Convolution-based sharpening.
    """
    image = normalize_to_uint8(
        image
    )

    strength = float(
        strength
    )

    base_kernel = np.array(
        [
            [-1, -1, -1],
            [-1, 12, -1],
            [-1, -1, -1],
        ],
        dtype=np.float32,
    )

    identity = np.zeros(
        (3, 3),
        dtype=np.float32,
    )

    identity[1, 1] = 1.0

    kernel = (
        identity
        + strength
        * (base_kernel - identity)
    )

    result = cv2.filter2D(
        image.astype(
            np.float32
        ),
        -1,
        kernel,
        borderType=cv2.BORDER_REPLICATE,
    )

    return normalize_to_uint8(
        result
    )


# ============================================================
# SOBEL
# ============================================================

def sobel_edges(
    image,
    strength=1.0,
):
    """
    Sobel edge detection.
    """
    image = normalize_to_uint8(
        image
    )

    img = image.astype(
        np.float32
    )

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

    magnitude = cv2.magnitude(
        gx,
        gy,
    )

    magnitude *= float(
        strength
    )

    return normalize_to_uint8(
        magnitude
    )


# ============================================================
# LAPLACIAN
# ============================================================

def laplacian_enhancement(
    image,
    strength=0.5,
):
    """
    Laplacian-based detail enhancement.
    """
    image = normalize_to_uint8(
        image
    )

    img = image.astype(
        np.float32
    )

    lap = cv2.Laplacian(
        img,
        cv2.CV_32F,
        ksize=3,
    )

    result = (
        img
        - float(strength) * lap
    )

    return normalize_to_uint8(
        result
    )


# ============================================================
# CONVOLUTION KERNELS
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


def convolve_image(
    image,
    kernel_name,
    strength=1.0,
):
    image = normalize_to_uint8(
        image
    )

    kernel = (
        CONVOLUTION_KERNELS[
            kernel_name
        ].copy()
    )

    identity = np.zeros(
        (3, 3),
        dtype=np.float32,
    )

    identity[1, 1] = 1.0

    kernel = (
        identity
        + float(strength)
        * (kernel - identity)
    )

    result = cv2.filter2D(
        image.astype(
            np.float32
        ),
        -1,
        kernel,
        borderType=cv2.BORDER_REPLICATE,
    )

    return normalize_to_uint8(
        result
    )


# ============================================================
# MINIMUM / MAXIMUM
# ============================================================

def minimum_filter(
    image,
    radius,
):
    image = normalize_to_uint8(
        image
    )

    radius = max(
        1,
        int(radius),
    )

    size = radius * 2 + 1

    kernel = np.ones(
        (
            size,
            size,
        ),
        dtype=np.uint8,
    )

    return cv2.erode(
        image,
        kernel,
        borderType=cv2.BORDER_REPLICATE,
    )


def maximum_filter(
    image,
    radius,
):
    image = normalize_to_uint8(
        image
    )

    radius = max(
        1,
        int(radius),
    )

    size = radius * 2 + 1

    kernel = np.ones(
        (
            size,
            size,
        ),
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

def morphological_opening(
    image,
    radius,
):
    image = normalize_to_uint8(
        image
    )

    radius = max(
        1,
        int(radius),
    )

    size = radius * 2 + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            size,
            size,
        ),
    )

    return cv2.morphologyEx(
        image,
        cv2.MORPH_OPEN,
        kernel,
    )


def morphological_closing(
    image,
    radius,
):
    image = normalize_to_uint8(
        image
    )

    radius = max(
        1,
        int(radius),
    )

    size = radius * 2 + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            size,
            size,
        ),
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
    Background correction using morphological opening.
    """
    image = normalize_to_uint8(
        image
    )

    radius = max(
        3,
        int(radius),
    )

    if radius % 2 == 0:
        radius += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            radius,
            radius,
        ),
    )

    background = cv2.morphologyEx(
        image,
        cv2.MORPH_OPEN,
        kernel,
    )

    result = (
        image.astype(
            np.float32
        )
        - background.astype(
            np.float32
        )
        + 128.0
    )

    return normalize_to_uint8(
        result
    )


# ============================================================
# GAMMA
# ============================================================

def gamma_correction(
    image,
    gamma,
):
    image = normalize_to_uint8(
        image
    )

    gamma = float(gamma)

    if gamma <= 0:
        gamma = 1.0

    normalized = (
        image.astype(
            np.float32
        )
        / 255.0
    )

    result = np.power(
        normalized,
        gamma,
    )

    return np.clip(
        result * 255.0,
        0,
        255,
    ).astype(
        np.uint8
    )


# ============================================================
# DENTAL PRESETS
# ============================================================

def dental_preset(
    image,
    sharpening=0.8,
):
    """
    General dental enhancement.
    """
    result = enhance_contrast(
        image,
        saturated=0.35,
    )

    result = apply_clahe(
        result,
        block_size=100,
        bins=256,
        max_slope=2.0,
    )

    result = unsharp_mask(
        result,
        sigma=1.0,
        weight=sharpening,
    )

    return result


def endodontic_preset(
    image,
    clahe_slope=2.5,
    usm_sigma=0.9,
    usm_weight=1.8,
):
    """
    Endodontic enhancement preset.
    """
    result = apply_clahe(
        image,
        block_size=80,
        bins=256,
        max_slope=clahe_slope,
    )

    result = gaussian_blur(
        result,
        sigma=0.45,
    )

    result = unsharp_mask(
        result,
        sigma=usm_sigma,
        weight=usm_weight,
    )

    return result


def perio_bone_preset(
    image,
    contrast=0.35,
    usm_sigma=1.5,
    usm_weight=1.1,
):
    """
    Periodontal and bone enhancement preset.
    """
    result = enhance_contrast(
        image,
        saturated=contrast,
    )

    result = apply_clahe(
        result,
        block_size=100,
        bins=256,
        max_slope=2.2,
    )

    result = unsharp_mask(
        result,
        sigma=usm_sigma,
        weight=usm_weight,
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
        help=(
            "PNG, JPG, TIFF and DICOM "
            "files are supported."
        ),
    )

    st.divider()

    st.header("Processing mode")

    mode = st.selectbox(
        "Choose operation",
        [
            "Dental Enhancement",
            "Endodontic Preset",
            "Perio / Bone Preset",
            "Enhance Contrast",
            "Histogram Equalization",
            "CLAHE",
            "Gaussian Blur",
            "Median Filter",
            "Mean Filter",
            "Unsharp Mask",
            "Sharpen",
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
# NO IMAGE
# ============================================================

if uploaded is None:

    st.info(
        "Upload a dental radiograph "
        "from the sidebar to begin."
    )

    st.markdown(
        """
        ### Available operations

        **Contrast**
        - Enhance Contrast
        - Histogram Equalization
        - CLAHE

        **Noise reduction**
        - Gaussian Blur
        - Median Filter
        - Mean Filter

        **Sharpening**
        - Unsharp Mask
        - Sharpen
        - Laplacian Enhancement

        **Edges and filtering**
        - Find Edges / Sobel
        - Convolve

        **Morphology**
        - Minimum
        - Maximum
        - Morphological Opening
        - Morphological Closing

        **Background**
        - Subtract Background

        **Intensity**
        - Gamma Correction

        **Dental presets**
        - Dental Enhancement
        - Endodontic Preset
        - Perio / Bone Preset
        """
    )

    st.stop()


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    original = load_uploaded_image(
        uploaded
    )

except Exception as exc:

    st.error(
        f"Could not load the image: {exc}"
    )

    st.stop()


original = normalize_to_uint8(
    original
)


# ============================================================
# MODE-SPECIFIC CONTROLS
# ============================================================

with st.sidebar:

    if mode == "Dental Enhancement":

        st.subheader(
            "Dental Enhancement"
        )

        st.caption(
            "General enhancement for dental radiographs."
        )

        preset_strength = st.slider(
            "Sharpening strength",
            0.0,
            2.0,
            0.8,
            0.05,
        )


    elif mode == "Endodontic Preset":

        st.subheader(
            "Endodontic"
        )

        st.caption(
            "Enhancement focused on fine endodontic detail."
        )

        clahe_slope = st.slider(
            "Contrast strength",
            1.0,
            5.0,
            2.5,
            0.1,
        )

        usm_sigma = st.slider(
            "Sharpening Sigma",
            0.3,
            2.0,
            0.9,
            0.05,
        )

        usm_weight = st.slider(
            "Sharpening strength",
            0.0,
            3.0,
            1.8,
            0.05,
        )


    elif mode == "Perio / Bone Preset":

        st.subheader(
            "Perio / Bone"
        )

        st.caption(
            "Enhancement focused on periodontal and osseous detail."
        )

        contrast = st.slider(
            "Contrast strength",
            0.05,
            2.0,
            0.35,
            0.05,
        )

        usm_sigma = st.slider(
            "Sharpening Sigma",
            0.5,
            3.0,
            1.5,
            0.05,
        )

        usm_weight = st.slider(
            "Sharpening strength",
            0.0,
            2.5,
            1.1,
            0.05,
        )


    elif mode == "Enhance Contrast":

        st.subheader(
            "Enhance Contrast"
        )

        saturated = st.slider(
            "Saturated pixels (%)",
            0.01,
            5.0,
            0.35,
            0.01,
        )


    elif mode == "Histogram Equalization":

        st.subheader(
            "Histogram Equalization"
        )

        st.caption(
            "Global histogram-based contrast enhancement."
        )


    elif mode == "CLAHE":

        st.subheader(
            "CLAHE"
        )

        st.caption(
            "Local contrast enhancement."
        )

        block_size = st.slider(
            "Block size",
            16,
            256,
            100,
            1,
        )

        bins = st.select_slider(
            "Histogram bins",
            options=[
                64,
                128,
                256,
            ],
            value=256,
        )

        max_slope = st.slider(
            "Contrast limit",
            1.0,
            8.0,
            2.5,
            0.1,
        )


    elif mode == "Gaussian Blur":

        st.subheader(
            "Gaussian Blur"
        )

        sigma = st.slider(
            "Sigma",
            0.1,
            10.0,
            1.0,
            0.1,
        )


    elif mode == "Median Filter":

        st.subheader(
            "Median Filter"
        )

        kernel = st.select_slider(
            "Kernel size",
            options=[
                3,
                5,
                7,
                9,
                11,
            ],
            value=3,
        )


    elif mode == "Mean Filter":

        st.subheader(
            "Mean Filter"
        )

        radius = st.slider(
            "Radius",
            1,
            10,
            1,
        )


    elif mode == "Unsharp Mask":

        st.subheader(
            "Unsharp Mask"
        )

        st.caption(
            "Enhances fine image detail and edge definition."
        )

        sigma = st.slider(
            "Sigma",
            0.1,
            5.0,
            1.0,
            0.05,
        )

        weight = st.slider(
            "Strength",
            0.0,
            5.0,
            0.6,
            0.05,
        )


    elif mode == "Sharpen":

        st.subheader(
            "Sharpen"
        )

        strength = st.slider(
            "Strength",
            0.0,
            2.0,
            1.0,
            0.05,
        )


    elif mode == "Find Edges / Sobel":

        st.subheader(
            "Find Edges / Sobel"
        )

        strength = st.slider(
            "Edge strength",
            0.1,
            3.0,
            1.0,
            0.05,
        )


    elif mode == "Laplacian Enhancement":

        st.subheader(
            "Laplacian Enhancement"
        )

        strength = st.slider(
            "Strength",
            0.0,
            2.0,
            0.5,
            0.05,
        )


    elif mode == "Convolve":

        st.subheader(
            "Convolve"
        )

        kernel_name = st.selectbox(
            "Kernel",
            list(
                CONVOLUTION_KERNELS.keys()
            ),
        )

        strength = st.slider(
            "Kernel strength",
            0.0,
            2.0,
            1.0,
            0.05,
        )


    elif mode == "Minimum":

        st.subheader(
            "Minimum"
        )

        radius = st.slider(
            "Radius",
            1,
            10,
            1,
        )


    elif mode == "Maximum":

        st.subheader(
            "Maximum"
        )

        radius = st.slider(
            "Radius",
            1,
            10,
            1,
        )


    elif mode == "Morphological Opening":

        st.subheader(
            "Morphological Opening"
        )

        radius = st.slider(
            "Radius",
            1,
            10,
            2,
        )


    elif mode == "Morphological Closing":

        st.subheader(
            "Morphological Closing"
        )

        radius = st.slider(
            "Radius",
            1,
            10,
            2,
        )


    elif mode == "Subtract Background":

        st.subheader(
            "Subtract Background"
        )

        background_radius = st.slider(
            "Background radius",
            5,
            150,
            25,
            1,
        )


    elif mode == "Gamma Correction":

        st.subheader(
            "Gamma Correction"
        )

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


if mode == "Dental Enhancement":

    processed = dental_preset(
        processed,
        sharpening=preset_strength,
    )


elif mode == "Endodontic Preset":

    processed = endodontic_preset(
        processed,
        clahe_slope=clahe_slope,
        usm_sigma=usm_sigma,
        usm_weight=usm_weight,
    )


elif mode == "Perio / Bone Preset":

    processed = perio_bone_preset(
        processed,
        contrast=contrast,
        usm_sigma=usm_sigma,
        usm_weight=usm_weight,
    )


elif mode == "Enhance Contrast":

    processed = enhance_contrast(
        processed,
        saturated=saturated,
    )


elif mode == "Histogram Equalization":

    processed = histogram_equalization(
        processed
    )


elif mode == "CLAHE":

    processed = apply_clahe(
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

    processed = unsharp_mask(
        processed,
        sigma=sigma,
        weight=weight,
    )


elif mode == "Sharpen":

    processed = sharpen(
        processed,
        strength=strength,
    )


elif mode == "Find Edges / Sobel":

    processed = sobel_edges(
        processed,
        strength=strength,
    )


elif mode == "Laplacian Enhancement":

    processed = laplacian_enhancement(
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

    processed = morphological_opening(
        processed,
        radius=radius,
    )


elif mode == "Morphological Closing":

    processed = morphological_closing(
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

st.markdown(
    f'<div class="operation-title">'
    f'{mode}'
    f'</div>',
    unsafe_allow_html=True,
)

display_original = resize_for_display(
    original
)

display_processed = resize_for_display(
    processed
)

col1, col2 = st.columns(
    2,
    gap="medium",
)

with col1:

    st.markdown(
        "### Original"
    )

    st.image(
        display_original,
        use_container_width=True,
        clamp=True,
    )


with col2:

    st.markdown(
        "### Processed"
    )

    st.image(
        display_processed,
        use_container_width=True,
        clamp=True,
    )


# ============================================================
# IMAGE INFORMATION
# ============================================================

st.divider()

info1, info2 = st.columns(2)

with info1:

    st.markdown(
        """
        <div class="info-box">
            <div class="info-label">
                Width
            </div>
            <div class="info-value">
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"{original.shape[1]} px"
    )

    st.markdown(
        "</div></div>",
        unsafe_allow_html=True,
    )


with info2:

    st.markdown(
        """
        <div class="info-box">
            <div class="info-label">
                Height
            </div>
            <div class="info-value">
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"{original.shape[0]} px"
    )

    st.markdown(
        "</div></div>",
        unsafe_allow_html=True,
    )


# ============================================================
# EXPORT
# ============================================================

st.divider()

st.subheader(
    "Export"
)

png_bytes = image_to_png_bytes(
    processed
)

base_name = Path(
    uploaded.name
).stem

safe_mode = (
    mode
    .lower()
    .replace(" ", "_")
    .replace("/", "-")
)

st.download_button(
    label="⬇️ Download processed PNG",
    data=png_bytes,
    file_name=(
        f"{base_name}_{safe_mode}.png"
    ),
    mime="image/png",
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#888;
        font-size:0.8rem;
        margin-top:2rem;
    ">
        Dental Image Processor ·
        Tasos Dimitrakopoulos · 2026
    </div>
    """,
    unsafe_allow_html=True,
)
```

And use this as your `requirements.txt`:

```text
streamlit
numpy
opencv-python-headless
Pillow
pydicom
```
