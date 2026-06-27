import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
from difflib import SequenceMatcher
import tempfile
import os
import io
import time
import shutil

# ─── Auto-detect Tesseract path ─────────────────────────────────────────────
def _find_tesseract() -> str:
    candidates = [
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        shutil.which("tesseract") or "",
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return "tesseract"

pytesseract.pytesseract.tesseract_cmd = _find_tesseract()

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="مستخرج الترجمة المدمجة",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="auto",
)

# ─── Custom CSS (Changa font + brand colors + responsive) ────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Changa:wght@300;400;500;600;700&display=swap');

/* ── Base ──────────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Changa', sans-serif !important;
    direction: rtl;
}

.stApp {
    background-color: #0f2532;
}

/* ── Sidebar ───────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #1A3A4A !important;
    border-right: 2px solid #2C6B7F;
    min-width: 260px !important;
}
[data-testid="stSidebar"] * {
    color: #e8f4f8 !important;
    font-family: 'Changa', sans-serif !important;
}

/* ── Typography ────────────────────────────────────────────────────────────── */
h1, h2, h3, h4 {
    font-family: 'Changa', sans-serif !important;
    color: #D4A24E !important;
    font-weight: 700;
    line-height: 1.3;
}
h1 { font-size: clamp(1.4rem, 4vw, 2.2rem) !important; }
h2 { font-size: clamp(1.1rem, 3vw, 1.6rem) !important; }
h3 { font-size: clamp(1rem, 2.5vw, 1.3rem) !important; }

p, span, label, div { color: #e8f4f8; }

/* ── Main content padding – tighter on small screens ───────────────────────── */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: clamp(0.75rem, 3vw, 3rem) !important;
    padding-right: clamp(0.75rem, 3vw, 3rem) !important;
    max-width: 100% !important;
}

/* ── Buttons ───────────────────────────────────────────────────────────────── */
.stButton > button {
    background-color: #2C6B7F !important;
    color: #ffffff !important;
    font-family: 'Changa', sans-serif !important;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 1.5rem;
    font-size: clamp(0.9rem, 2.5vw, 1.05rem);
    transition: all 0.2s ease;
    width: 100%;
    min-height: 48px;        /* touch-friendly */
}
.stButton > button:hover {
    background-color: #1A3A4A !important;
    border: 1px solid #D4A24E !important;
    color: #D4A24E !important;
}
.stButton > button:active {
    transform: scale(0.97);
}

/* ── Download button ───────────────────────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    background-color: #D4A24E !important;
    color: #1A3A4A !important;
    font-family: 'Changa', sans-serif !important;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 1.5rem;
    width: 100%;
    min-height: 52px;
    font-size: clamp(1rem, 2.5vw, 1.15rem);
}
[data-testid="stDownloadButton"] > button:hover {
    background-color: #c49040 !important;
}

/* ── Progress bar ──────────────────────────────────────────────────────────── */
.stProgress > div > div > div {
    background-color: #D4A24E !important;
}

/* ── Alerts / info boxes ───────────────────────────────────────────────────── */
.stAlert {
    border-radius: 10px;
    font-family: 'Changa', sans-serif !important;
    font-size: clamp(0.85rem, 2vw, 1rem);
}

/* ── Select / Slider labels ────────────────────────────────────────────────── */
.stSelectbox label, .stFileUploader label, .stSlider label {
    color: #D4A24E !important;
    font-family: 'Changa', sans-serif !important;
    font-weight: 600;
    font-size: clamp(0.85rem, 2vw, 1rem);
}

/* ── File uploader ─────────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border-radius: 12px;
    background-color: #1A3A4A;
    border: 2px dashed #2C6B7F;
    padding: 0.5rem;
}

/* ── Text area ─────────────────────────────────────────────────────────────── */
.stTextArea textarea {
    background-color: #1A3A4A !important;
    color: #e8f4f8 !important;
    font-family: 'Courier New', monospace !important;
    border: 1px solid #2C6B7F;
    border-radius: 8px;
    font-size: clamp(0.75rem, 1.8vw, 0.9rem);
}

/* ── Metric cards ──────────────────────────────────────────────────────────── */
.metric-card {
    background-color: #1A3A4A;
    border: 1px solid #2C6B7F;
    border-radius: 12px;
    padding: clamp(0.6rem, 2vw, 1rem);
    text-align: center;
    margin-bottom: 0.75rem;
}
.metric-card .value {
    font-size: clamp(1.2rem, 4vw, 2rem);
    font-weight: 700;
    color: #D4A24E;
    word-break: break-word;
}
.metric-card .label {
    font-size: clamp(0.75rem, 2vw, 0.9rem);
    color: #a0c4d4;
    margin-top: 4px;
}

/* ── Section divider ───────────────────────────────────────────────────────── */
.section-divider {
    border-top: 1px solid #2C6B7F;
    margin: 1rem 0;
}

/* ── Live text box ─────────────────────────────────────────────────────────── */
.live-text-box {
    background-color: #1A3A4A;
    border: 1px solid #D4A24E;
    border-radius: 10px;
    padding: clamp(0.6rem, 2vw, 1rem);
    min-height: 70px;
    font-size: clamp(0.9rem, 2.5vw, 1.1rem);
    color: #ffffff;
    direction: rtl;
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
    word-break: break-word;
}

/* ── Landing hero ──────────────────────────────────────────────────────────── */
.landing-hero {
    text-align: center;
    padding: clamp(1.5rem, 5vw, 3rem) 0;
}
.landing-hero .hero-icon {
    font-size: clamp(3rem, 10vw, 5rem);
}
.landing-hero h3 {
    font-size: clamp(1.1rem, 3.5vw, 1.6rem) !important;
    margin-top: 0.5rem;
}
.landing-hero p {
    font-size: clamp(0.85rem, 2.2vw, 1rem);
    color: #a0c4d4;
    max-width: 520px;
    margin: 0.75rem auto 0;
    line-height: 1.6;
}

/* ── Steps list ────────────────────────────────────────────────────────────── */
.steps-list {
    font-size: clamp(0.82rem, 2vw, 0.95rem);
    line-height: 1.9;
}

/* ── RESPONSIVE BREAKPOINTS ────────────────────────────────────────────────── */

/* Tablet  (≤ 1024 px) */
@media (max-width: 1024px) {
    [data-testid="stSidebar"] {
        min-width: 230px !important;
    }
    .block-container {
        padding-left: clamp(0.5rem, 2vw, 1.5rem) !important;
        padding-right: clamp(0.5rem, 2vw, 1.5rem) !important;
    }
}

/* Mobile (≤ 768 px) */
@media (max-width: 768px) {
    /* Stack columns vertically */
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 0.5rem !important;
    }
    [data-testid="stHorizontalBlock"] > div {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }

    /* Compact padding */
    .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-top: 1rem !important;
    }

    /* Smaller metric cards */
    .metric-card {
        padding: 0.5rem;
        border-radius: 10px;
    }

    /* Sidebar full-width when expanded on mobile */
    [data-testid="stSidebar"] {
        min-width: 100vw !important;
        max-width: 100vw !important;
    }

    /* Bigger touch areas for sliders */
    [data-testid="stSlider"] input[type="range"] {
        height: 28px;
    }

    /* Buttons fill the screen */
    .stButton > button,
    [data-testid="stDownloadButton"] > button {
        min-height: 54px;
        font-size: 1rem;
        border-radius: 12px;
    }

    .live-text-box {
        min-height: 60px;
        font-size: 0.95rem;
    }
}

/* Small mobile (≤ 480 px) */
@media (max-width: 480px) {
    h1 { font-size: 1.25rem !important; }
    h3 { font-size: 0.95rem !important; }

    .block-container {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }

    .metric-card .value { font-size: 1.1rem; }
    .metric-card .label { font-size: 0.72rem; }

    .stAlert { font-size: 0.82rem; }
}
</style>
""", unsafe_allow_html=True)

# ─── Language map ────────────────────────────────────────────────────────────
LANGUAGE_MAP = {
    "العربية": "ara",
    "الإنجليزية": "eng",
    "الروسية": "rus",
    "التركية": "tur",
    "الهندية": "hin",
    "الأردية": "urd",
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def format_srt_time(seconds: float) -> str:
    """Convert float seconds to SRT timestamp HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def similarity(a: str, b: str) -> float:
    """Return SequenceMatcher similarity ratio."""
    return SequenceMatcher(None, a.strip(), b.strip()).ratio()


def extract_text_from_frame(frame: np.ndarray, lang_code: str) -> str:
    """Crop bottom 22 %, convert to grayscale, run Tesseract OCR."""
    h = frame.shape[0]
    crop_y = int(h * 0.78)
    cropped = frame[crop_y:, :]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pil_img = Image.fromarray(thresh)
    config = "--psm 6"
    try:
        text = pytesseract.image_to_string(pil_img, lang=lang_code, config=config)
        return text.strip()
    except Exception:
        return ""


def get_cropped_preview(frame: np.ndarray) -> Image.Image:
    """Return the bottom-22 % crop as PIL image for preview."""
    h = frame.shape[0]
    crop_y = int(h * 0.78)
    cropped = frame[crop_y:, :]
    return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))


def build_srt(subtitles: list) -> str:
    """Build SRT string from list of (index, start, end, text) tuples."""
    lines = []
    for idx, (start, end, text) in enumerate(subtitles, 1):
        lines.append(str(idx))
        lines.append(f"{format_srt_time(start)} --> {format_srt_time(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 مستخرج الترجمة المدمجة")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### ⚙️ إعدادات المعالجة")
    selected_lang_name = st.selectbox(
        "لغة الترجمة في الفيلم",
        list(LANGUAGE_MAP.keys()),
        index=0,
    )
    lang_code = LANGUAGE_MAP[selected_lang_name]

    similarity_threshold = st.slider(
        "عتبة التشابه لدمج الجمل (%)",
        min_value=50,
        max_value=95,
        value=75,
        step=5,
        help="إذا كان التشابه بين الجملة الحالية والسابقة أعلى من هذه النسبة، سيتم دمجهما في سطر واحد",
    ) / 100.0

    frames_per_sec = st.slider(
        "عدد الإطارات المفحوصة في الثانية",
        min_value=1,
        max_value=4,
        value=2,
        step=1,
        help="القيمة الموصى بها: 2 إطار/ثانية لتوازن الدقة والسرعة",
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### ℹ️ معلومات")
    st.info(
        "يستخدم هذا التطبيق **Tesseract OCR** لاستخراج الترجمة المدمجة (Hardsubs) "
        "من الفيديوهات وتحويلها إلى ملفات **SRT** قابلة للتعديل.",
        icon="🤖",
    )
    st.markdown(
        """
        <div class="steps-list">

        **الخطوات:**
        1. ارفع ملف الفيديو
        2. اختر لغة الترجمة
        3. اضغط **ابدأ المعالجة**
        4. انتظر انتهاء التحليل
        5. حمّل ملف SRT

        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Main area ───────────────────────────────────────────────────────────────
st.markdown("# 🎬 مستخرج الترجمة المدمجة (Hardsubs → SRT)")
st.markdown(
    "ارفع فيديو يحتوي على ترجمة مدمجة، وسيقوم الذكاء الاصطناعي بقراءتها وتحويلها "
    "إلى ملف **SRT** جاهز للتنزيل والتعديل."
)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📂 ارفع ملف الفيديو",
    type=["mp4", "avi", "mkv", "mov", "wmv", "flv", "webm"],
    help="الصيغ المدعومة: MP4, AVI, MKV, MOV, WMV, FLV, WebM",
)

if uploaded_file is not None:
    # Validate file
    file_ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    if file_ext not in ["mp4", "avi", "mkv", "mov", "wmv", "flv", "webm"]:
        st.error("❌ صيغة الملف غير مدعومة. يرجى رفع ملف فيديو صالح.")
        st.stop()

    # Show file info
    file_size_mb = uploaded_file.size / (1024 * 1024)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            f'<div class="metric-card"><div class="value">📄</div>'
            f'<div class="label">{uploaded_file.name}</div></div>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f'<div class="metric-card"><div class="value">{file_size_mb:.1f} MB</div>'
            f'<div class="label">حجم الملف</div></div>',
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            f'<div class="metric-card"><div class="value">{selected_lang_name}</div>'
            f'<div class="label">لغة الترجمة</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if st.button("🚀 ابدأ المعالجة", type="primary"):
        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                st.error("❌ تعذّر فتح الملف. تأكد من أن الفيديو غير تالف وبصيغة مدعومة.")
                os.unlink(tmp_path)
                st.stop()

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 25.0
            duration_sec = total_frames / fps

            # How many frames to skip between checks
            frame_interval = max(1, int(fps / frames_per_sec))

            st.info(
                f"⏱️ مدة الفيديو: **{int(duration_sec // 60)}د {int(duration_sec % 60)}ث** | "
                f"معدل الإطارات: **{fps:.1f} fps** | "
                f"سيتم فحص إطار كل **{frame_interval}** إطار",
                icon="📊",
            )

            # ── Live preview layout ──
            st.markdown("### 🔴 المعالجة الحية")
            col_frame, col_text = st.columns([1, 1])

            with col_frame:
                st.markdown("**📷 الإطار المقصوص الحالي (منطقة الترجمة)**")
                frame_placeholder = st.empty()

            with col_text:
                st.markdown("**📝 النص المستخرج في الوقت الفعلي**")
                text_placeholder = st.empty()

            progress_bar = st.progress(0)
            status_text = st.empty()

            subtitles = []
            current_sub = None
            frame_idx = 0
            processed = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    current_time = frame_idx / fps
                    progress = min(frame_idx / max(total_frames, 1), 1.0)
                    progress_bar.progress(progress)
                    status_text.markdown(
                        f"⏳ جاري المعالجة... **{format_srt_time(current_time)}** "
                        f"({int(progress * 100)}%)"
                    )

                    # Show cropped frame preview
                    preview_img = get_cropped_preview(frame)
                    frame_placeholder.image(
                        preview_img,
                        caption=f"⏱ {format_srt_time(current_time)}",
                        use_container_width=True,
                    )

                    # OCR
                    text = extract_text_from_frame(frame, lang_code)

                    # Live text display
                    display_text = text if text else "— لا يوجد نص —"
                    text_placeholder.markdown(
                        f'<div class="live-text-box">{display_text}</div>',
                        unsafe_allow_html=True,
                    )

                    # Subtitle merging logic
                    if text:
                        if current_sub is None:
                            current_sub = {
                                "start": current_time,
                                "end": current_time,
                                "text": text,
                            }
                        else:
                            sim = similarity(text, current_sub["text"])
                            if sim >= similarity_threshold:
                                current_sub["end"] = current_time
                            else:
                                subtitles.append(
                                    (current_sub["start"], current_sub["end"], current_sub["text"])
                                )
                                current_sub = {
                                    "start": current_time,
                                    "end": current_time,
                                    "text": text,
                                }
                    else:
                        if current_sub is not None:
                            subtitles.append(
                                (current_sub["start"], current_sub["end"], current_sub["text"])
                            )
                            current_sub = None

                    processed += 1

                frame_idx += 1

            # Flush last subtitle
            if current_sub is not None:
                subtitles.append(
                    (current_sub["start"], current_sub["end"], current_sub["text"])
                )

            cap.release()
            os.unlink(tmp_path)

            progress_bar.progress(1.0)
            status_text.empty()

        except Exception as e:
            st.error(f"❌ حدث خطأ أثناء المعالجة: {str(e)}")
            try:
                cap.release()
                os.unlink(tmp_path)
            except Exception:
                pass
            st.stop()

        # ── Results ──
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        if not subtitles:
            st.warning(
                "⚠️ لم يتم اكتشاف أي نص. تأكد من أن الفيديو يحتوي على ترجمة مدمجة واضحة، "
                "وأنك اخترت اللغة الصحيحة.",
                icon="🔍",
            )
        else:
            srt_content = build_srt(subtitles)

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown(
                    f'<div class="metric-card"><div class="value">{len(subtitles)}</div>'
                    f'<div class="label">جملة مستخرجة</div></div>',
                    unsafe_allow_html=True,
                )
            with col_r2:
                st.markdown(
                    f'<div class="metric-card"><div class="value">{processed}</div>'
                    f'<div class="label">إطار تم فحصه</div></div>',
                    unsafe_allow_html=True,
                )

            st.success(
                f"✅ تمت المعالجة بنجاح! تم استخراج **{len(subtitles)}** سطر ترجمة.",
                icon="🎉",
            )

            # Download button
            srt_bytes = srt_content.encode("utf-8")
            st.download_button(
                label="⬇️ تنزيل ملف SRT",
                data=srt_bytes,
                file_name="extracted_subtitles.srt",
                mime="text/plain",
            )

            # Preview
            st.markdown("### 👁️ معاينة ملف SRT")
            st.text_area(
                "محتوى الملف (يمكن نسخه يدوياً)",
                value=srt_content,
                height=400,
            )

else:
    # Landing state
    st.markdown(
        """
        <div class="landing-hero">
            <div class="hero-icon">🎬</div>
            <h3>ارفع ملف فيديو للبدء</h3>
            <p>
                يدعم التطبيق الفيديوهات التي تحتوي على ترجمة مدمجة (Hardsubs)
                بـ 6 لغات مختلفة. قم بتحديد إعدادات المعالجة من القائمة الجانبية
                ثم ارفع الفيديو.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
