from skimage.restoration import denoise_tv_chambolle
import io
import math from pathlib 
import Path
import cv2
import numpy as np
import pydicom
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Dental Image Processor",
    page_icon="🦷",
    layout="wide",
)

st.title("🦷 Dental Image Processor")

st.caption(
    "CLAHE • Median Filter • Standard Sharpening • "
    "Unsharp Masking • DICOM • Pixel/mm Measurement"
)


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

def normalize_to_uint8(array):
    """
    Convert arbitrary grayscale data to uint8.
    """

    array = np.asarray(
        array,
        dtype=np.float32,
    )

    finite = np.isfinite(array)

    if not np.any(finite):

        return np.zeros(
            array.shape,
            dtype=np.uint8,
        )

    minimum = np.min(
        array[finite]
    )

    maximum = np.max(
        array[finite]
    )

    if maximum <= minimum:

        return np.zeros(
            array.shape,
            dtype=np.uint8,
        )

    normalized = (
        (array - minimum)
        / (maximum - minimum)
        * 255.0
    )

    return np.clip(
        normalized,
        0,
        255,
    ).astype(
        np.uint8
    )


# ============================================================
# DICOM LOADING
# ============================================================

def load_dicom(file):

    ds = pydicom.dcmread(
        file,
        force=True,
    )

    # --------------------------------------------------------
    # Read pixel data
    # --------------------------------------------------------

    try:

        image = ds.pixel_array.astype(
            np.float32
        )

    except Exception as error:

        raise RuntimeError(
            "This DICOM contains compressed pixel data "
            "that the current Playground environment "
            "cannot decode.\n\n"
            "If this is a JPEG Lossless DICOM, the "
            "normal Streamlit deployment should use "
            "pylibjpeg/gdcm for decoding.\n\n"
            f"Original error:\n{error}"
        ) from error

    # --------------------------------------------------------
    # Rescale slope/intercept
    # --------------------------------------------------------

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

    image = (
        image * slope
        + intercept
    )

    # --------------------------------------------------------
    # MONOCHROME1
    # --------------------------------------------------------

    photometric = str(
        getattr(
            ds,
            "PhotometricInterpretation",
            "MONOCHROME2",
        )
    )

    if photometric == "MONOCHROME1":

        image = (
            np.max(image)
            - image
        )

    # --------------------------------------------------------
    # DICOM Window Center / Width
    # --------------------------------------------------------

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

            if hasattr(
                window_center,
                "__len__",
            ):

                center = float(
                    window_center[0]
                )

            else:

                center = float(
                    window_center
                )

            if hasattr(
                window_width,
                "__len__",
            ):

                width_value = float(
                    window_width[0]
                )

            else:

                width_value = float(
                    window_width
                )

            if width_value > 1:

                low = (
                    center
                    - width_value / 2
                )

                high = (
                    center
                    + width_value / 2
                )

                image = np.clip(
                    image,
                    low,
                    high,
                )

        except Exception:
            pass

    # --------------------------------------------------------
    # Normalize to uint8
    # --------------------------------------------------------

    image = normalize_to_uint8(
        image
    )

    # --------------------------------------------------------
    # Pixel spacing
    # --------------------------------------------------------

    pixel_spacing = getattr(
        ds,
        "PixelSpacing",
        None,
    )

    spacing = None

    if (
        pixel_spacing is not None
        and len(pixel_spacing) >= 2
    ):

        try:

            spacing = (
                float(
                    pixel_spacing[0]
                ),
                float(
                    pixel_spacing[1]
                ),
            )

        except Exception:

            spacing = None

    return (
        image,
        ds,
        spacing,
    )


# ============================================================
# REGULAR IMAGE LOADING
# ============================================================

def load_regular_image(file):

    data = file.read()

    image = Image.open(
        io.BytesIO(data)
    ).convert("L")

    return (
        np.array(image),
        None,
        None,
    )


def load_image(file):

    filename = file.name.lower()

    if filename.endswith(
        ".dcm"
    ):

        return load_dicom(
            file
        )

    try:

        file.seek(0)

        return load_regular_image(
            file
        )

    except Exception:

        file.seek(0)

        return load_dicom(
            file
        )


# ============================================================
# IMAGE PROCESSING
# ============================================================

def process_image(
    image,

    # Median
    use_median,
    median_kernel,

    # CLAHE
    use_clahe,
    clip_limit,
    tile_size,

    # Standard sharpening
    use_sharpen,
    sharpen_strength,

    # USM
    use_usm,
    usm_sigma,
    usm_amount,
    usm_threshold,
):

    result = image.copy()

    # --------------------------------------------------------
    # Make absolutely sure OpenCV receives uint8
    # --------------------------------------------------------

    if result.dtype != np.uint8:

        result = normalize_to_uint8(
            result
        )

    # ========================================================
    # MEDIAN FILTER
    # ========================================================

    if use_median:

        result = cv2.medianBlur(
            result,
            int(
                median_kernel
            ),
        )

    # ========================================================
    # CLAHE
    # ========================================================

    if use_clahe:

        clahe = cv2.createCLAHE(
            clipLimit=float(
                clip_limit
            ),
            tileGridSize=(
                int(
                    tile_size
                ),
                int(
                    tile_size
                ),
            ),
        )

        result = clahe.apply(
            result
        )

    # ========================================================
    # STANDARD SHARPENING
    # ========================================================
    #
    # This is your ORIGINAL sharpening method.
    #
    # It remains completely independent from USM.
    #
    # ========================================================

    if use_sharpen:

        sharpening_kernel = np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ],
            dtype=np.float32,
        )

        identity = np.zeros(
            (
                3,
                3,
            ),
            dtype=np.float32,
        )

        identity[1, 1] = 1.0

        kernel = (
            identity
            + float(
                sharpen_strength
            )
            * (
                sharpening_kernel
                - identity
            )
        )

        result = cv2.filter2D(
            result,
            -1,
            kernel,
        )

        result = np.clip(
            result,
            0,
            255,
        ).astype(
            np.uint8
        )

    # ========================================================
    # UNSHARP MASKING (USM)
    # ========================================================
    #
    # USM:
    #
    # sharpened =
    # original + amount * (original - blurred)
    #
    # ========================================================

    if use_usm:

        # ----------------------------------------------------
        # Gaussian blur
        # ----------------------------------------------------

        blurred = cv2.GaussianBlur(
            result,
            (
                0,
                0,
            ),
            sigmaX=float(
                usm_sigma
            ),
            sigmaY=float(
                usm_sigma
            ),
        )

        # ----------------------------------------------------
        # Difference / high-frequency mask
        # ----------------------------------------------------

        mask = (
            result.astype(
                np.float32
            )
            -
            blurred.astype(
                np.float32
            )
        )

        # ----------------------------------------------------
        # Threshold
        #
        # Small differences can be interpreted as noise.
        # Setting a threshold to 0 disables thresholding.
        # ----------------------------------------------------

        if float(
            usm_threshold
        ) > 0:

            mask[
                np.abs(mask)
                < float(
                    usm_threshold
                )
            ] = 0.0

        # ----------------------------------------------------
        # Apply USM
        # ----------------------------------------------------

        sharpened = (
            result.astype(
                np.float32
            )
            +
            float(
                usm_amount
            )
            * mask
        )

        result = np.clip(
            sharpened,
            0,
            255,
        ).astype(
            np.uint8
        )

    return result


# ============================================================
# SESSION STATE
# ============================================================

if "image" not in st.session_state:

    st.session_state.image = None

if "filename" not in st.session_state:

    st.session_state.filename = None

if "dicom" not in st.session_state:

    st.session_state.dicom = None

if "spacing" not in st.session_state:

    st.session_state.spacing = None

if "calibration" not in st.session_state:

    st.session_state.calibration = None

if "file_signature" not in st.session_state:

    st.session_state.file_signature = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📂 Image")

    uploaded_file = st.file_uploader(
        "Drag & drop dental image",
        type=[
            "dcm",
            "png",
            "jpg",
            "jpeg",
            "tif",
            "tiff",
            "bmp",
        ],
    )

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    if uploaded_file is not None:

        file_signature = (
            uploaded_file.name,
            uploaded_file.size,
        )

        if (
            st.session_state.file_signature
            != file_signature
        ):

            try:

                (
                    image,
                    dicom,
                    spacing,
                ) = load_image(
                    uploaded_file
                )

                st.session_state.image = (
                    image
                )

                st.session_state.filename = (
                    uploaded_file.name
                )

                st.session_state.dicom = (
                    dicom
                )

                st.session_state.spacing = (
                    spacing
                )

                st.session_state.calibration = (
                    None
                )

                st.session_state.file_signature = (
                    file_signature
                )

                st.success(
                    f"Loaded: "
                    f"{uploaded_file.name}"
                )

            except Exception as error:

                st.error(
                    "Could not load image."
                )

                st.code(
                    str(error)
                )

    # ========================================================
    # PROCESSING
    # ========================================================

    if st.session_state.image is not None:

        st.divider()

        st.header("⚙️ Processing")

        # ====================================================
        # MEDIAN FILTER
        # ====================================================

        st.subheader(
            "Median Filter"
        )

        use_median = st.checkbox(
            "Enable Median Filter",
            value=False,
        )

        median_kernel = st.selectbox(
            "Median kernel",
            [
                3,
                5,
                7,
                9,
            ],
            disabled=not use_median,
        )

        # ====================================================
        # CLAHE
        # ====================================================

        st.subheader(
            "CLAHE"
        )

        use_clahe = st.checkbox(
            "Enable CLAHE",
            value=True,
        )

        clip_limit = st.slider(
            "CLAHE clip limit",
            0.1,
            10.0,
            2.0,
            0.1,
            disabled=not use_clahe,
        )

        tile_size = st.slider(
            "CLAHE tile size",
            2,
            32,
            8,
            1,
            disabled=not use_clahe,
        )

        # ====================================================
        # STANDARD SHARPENING
        # ====================================================

        st.subheader(
            "Standard Sharpening"
        )

        use_sharpen = st.checkbox(
            "Enable Standard Sharpening",
            value=False,
        )

        sharpen_strength = st.slider(
            "Sharpening strength",
            0.0,
            2.0,
            1.0,
            0.1,
            disabled=not use_sharpen,
        )

        # ====================================================
        # UNSHARP MASKING
        # ====================================================

        st.subheader(
            "Unsharp Masking (USM)"
        )

        use_usm = st.checkbox(
            "Enable USM",
            value=False,
        )

        usm_sigma = st.slider(
            "USM Gaussian sigma",
            0.1,
            5.0,
            1.0,
            0.1,
            disabled=not use_usm,
        )

        usm_amount = st.slider(
            "USM amount",
            0.0,
            5.0,
            1.0,
            0.1,
            disabled=not use_usm,
        )

        usm_threshold = st.slider(
            "USM threshold",
            0,
            50,
            5,
            1,
            disabled=not use_usm,
        )

        # ====================================================
        # MEASUREMENT
        # ====================================================

        st.divider()

        st.header(
            "📏 Measurement"
        )

        if st.session_state.spacing:

            sx, sy = (
                st.session_state.spacing
            )

            st.success(
                f"Row spacing: "
                f"{sx:.6f} mm/pixel\n\n"
                f"Column spacing: "
                f"{sy:.6f} mm/pixel"
            )

        else:

            st.warning(
                "No DICOM PixelSpacing detected."
            )

        st.subheader(
            "Manual calibration"
        )

        known_distance = st.number_input(
            "Reference distance (mm)",
            min_value=0.001,
            value=10.0,
            step=0.5,
        )

        if st.button(
            "Clear calibration"
        ):

            st.session_state.calibration = (
                None
            )

            st.rerun()


# ============================================================
# STOP IF NO IMAGE
# ============================================================

if st.session_state.image is None:

    st.info(
        "👆 Upload a dental image using the sidebar."
    )

    st.stop()


# ============================================================
# PROCESS
# ============================================================

original = (
    st.session_state.image
)

processed = process_image(
    original,

    # Median
    use_median,
    median_kernel,

    # CLAHE
    use_clahe,
    clip_limit,
    tile_size,

    # Standard sharpening
    use_sharpen,
    sharpen_strength,

    # USM
    use_usm,
    usm_sigma,
    usm_amount,
    usm_threshold,
)


# ============================================================
# DISPLAY SCALING
# ============================================================

height, width = (
    processed.shape[:2]
)

MAX_WIDTH = 1000

if width > MAX_WIDTH:

    scale = (
        MAX_WIDTH
        / float(width)
    )

else:

    scale = 1.0

display_width = max(
    1,
    int(
        round(
            width * scale
        )
    ),
)

display_height = max(
    1,
    int(
        round(
            height * scale
        )
    ),
)


original_display = cv2.resize(
    original,
    (
        display_width,
        display_height,
    ),
    interpolation=cv2.INTER_AREA,
)

processed_display = cv2.resize(
    processed,
    (
        display_width,
        display_height,
    ),
    interpolation=cv2.INTER_AREA,
)


# ============================================================
# TABS
# ============================================================

comparison_tab, measurement_tab, dicom_tab = (
    st.tabs(
        [
            "🖼️ Image",
            "📏 Measurement",
            "🏥 DICOM",
        ]
    )
)


# ============================================================
# IMAGE TAB
# ============================================================

with comparison_tab:

    left, right = st.columns(
        2
    )

    with left:

        st.subheader(
            "Original"
        )

        # IMPORTANT:
        # Use use_container_width=True.
        #
        # Do NOT use:
        #
        # width="stretch"
        #
        # because older Streamlit versions in
        # Playground can produce:
        #
        # TypeError:
        # '<=' not supported between
        # instances of 'str' and 'int'

        st.image(
            original_display,
            use_container_width=True,
        )

    with right:

        st.subheader(
            "Processed"
        )

        st.image(
            processed_display,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # Processing summary
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Current processing"
    )

    active_processing = []

    if use_median:

        active_processing.append(
            f"Median {median_kernel}×{median_kernel}"
        )

    if use_clahe:

        active_processing.append(
            "CLAHE"
        )

    if use_sharpen:

        active_processing.append(
            "Standard Sharpening"
        )

    if use_usm:

        active_processing.append(
            "Unsharp Masking (USM)"
        )

    if len(
        active_processing
    ) == 0:

        st.info(
            "No processing is enabled."
        )

    else:

        st.write(
            " → ".join(
                active_processing
            )
        )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    output = io.BytesIO()

    Image.fromarray(
        processed
    ).save(
        output,
        format="PNG",
    )

    st.download_button(
        "⬇️ Download processed image",
        output.getvalue(),
        file_name=(
            f"{Path(st.session_state.filename).stem}"
            "_processed.png"
        ),
        mime="image/png",
    )


# ============================================================
# MEASUREMENT TAB
# ============================================================

with measurement_tab:

    st.subheader(
        "📏 Point-to-point measurement"
    )

    st.write(
        "Draw a line between two points on the image."
    )

    canvas = st_canvas(
        fill_color=(
            "rgba(255, 0, 0, 0.1)"
        ),
        stroke_width=3,
        stroke_color="#ff0000",
        background_image=Image.fromarray(
            processed_display
        ),
        drawing_mode="line",
        height=int(
            display_height
        ),
        width=int(
            display_width
        ),
        update_streamlit=True,
        key="measurement_canvas",
    )

    # --------------------------------------------------------
    # Measurement
    # --------------------------------------------------------

    if canvas.json_data is not None:

        objects = (
            canvas.json_data.get(
                "objects",
                [],
            )
        )

        if len(objects) > 0:

            line = objects[-1]

            # Fabric.js coordinates
            x1 = float(
                line.get(
                    "x1",
                    0,
                )
            )

            y1 = float(
                line.get(
                    "y1",
                    0,
                )
            )

            x2 = float(
                line.get(
                    "x2",
                    0,
                )
            )

            y2 = float(
                line.get(
                    "y2",
                    0,
                )
            )

            scale_x = float(
                line.get(
                    "scaleX",
                    1,
                )
            )

            scale_y = float(
                line.get(
                    "scaleY",
                    1,
                )
            )

            dx_display = (
                (x2 - x1)
                * scale_x
            )

            dy_display = (
                (y2 - y1)
                * scale_y
            )

            display_distance = math.sqrt(
                dx_display ** 2
                +
                dy_display ** 2
            )

            # ------------------------------------------------
            # Convert display pixels to source pixels
            # ------------------------------------------------

            if scale > 0:

                source_distance = (
                    display_distance
                    / scale
                )

            else:

                source_distance = 0.0

            st.metric(
                "Distance",
                f"{source_distance:.2f} pixels",
            )

            # ------------------------------------------------
            # DICOM physical distance
            # ------------------------------------------------

            if st.session_state.spacing:

                sx, sy = (
                    st.session_state.spacing
                )

                dx = (
                    dx_display
                    / scale
                )

                dy = (
                    dy_display
                    / scale
                )

                # x corresponds to columns
                # and therefore column spacing = sy
                #
                # y corresponds to rows
                # and therefore row spacing = sx

                distance_mm = math.sqrt(
                    (dx * sy) ** 2
                    +
                    (dy * sx) ** 2
                )

                st.metric(
                    "Physical distance",
                    f"{distance_mm:.3f} mm",
                )

            # ------------------------------------------------
            # Manual calibration
            # ------------------------------------------------

            elif (
                st.session_state.calibration
            ):

                mm_per_pixel = float(
                    st.session_state.calibration
                )

                distance_mm = (
                    source_distance
                    *
                    mm_per_pixel
                )

                st.metric(
                    "Calibrated distance",
                    f"{distance_mm:.3f} mm",
                )

            # ------------------------------------------------
            # Calibration
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "Calibration"
            )

            st.write(
                "For a non-DICOM image, draw the line "
                "over an object with a known physical length."
            )

            if st.button(
                "Set calibration from this line"
            ):

                if source_distance > 0:

                    mm_per_pixel = (
                        float(
                            known_distance
                        )
                        /
                        float(
                            source_distance
                        )
                    )

                    st.session_state.calibration = (
                        mm_per_pixel
                    )

                    st.success(
                        "Calibration: "
                        f"{mm_per_pixel:.8f} mm/pixel"
                    )

                    st.rerun()

                else:

                    st.warning(
                        "Please draw a valid line first."
                    )


# ============================================================
# DICOM TAB
# ============================================================

with dicom_tab:

    st.subheader(
        "🏥 DICOM metadata"
    )

    ds = (
        st.session_state.dicom
    )

    if ds is None:

        st.info(
            "The uploaded image is not a DICOM file."
        )

    else:

        fields = [
            "PatientID",
            "StudyDate",
            "Modality",
            "Rows",
            "Columns",
            "BitsAllocated",
            "BitsStored",
            "HighBit",
            "PixelRepresentation",
            "PhotometricInterpretation",
            "PixelSpacing",
            "SliceThickness",
            "WindowCenter",
            "WindowWidth",
            "RescaleSlope",
            "RescaleIntercept",
            "SamplesPerPixel",
            "PlanarConfiguration",
            "NumberOfFrames",
        ]

        metadata = {}

        for field in fields:

            if hasattr(
                ds,
                field,
            ):

                metadata[field] = str(
                    getattr(
                        ds,
                        field,
                    )
                )

        st.json(
            metadata
        )

        st.warning(
            "Original DICOM pixel data is not "
            "overwritten. Processing is applied "
            "to a derived image."
        )
