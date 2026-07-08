import json
import os
import queue
import threading
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import (
    Flask, Response, jsonify, redirect,
    render_template, request, send_file, session, url_for,
)
from PIL import Image

from pipeline.page_renderer import PageRenderer
from pipeline.ocr import OCRProcessor
from pipeline.layout import build_lines
from pipeline.eraser import BackgroundEraser
from pipeline.translator import Translator
from pipeline.renderer import TextRenderer
from pipeline.assembler import PDFAssembler

load_dotenv()

FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD    = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs/web")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}

SUPPORTED_LANGUAGES = [
    "French", "German", "Dutch", "Spanish", "Italian",
    "Portuguese", "Polish", "Swedish", "Danish", "Norwegian",
    "Finnish", "Romanian", "Czech", "Slovak", "Hungarian",
    "Bulgarian", "Croatian", "Serbian", "Ukrainian", "Russian",
    "Greek", "Turkish", "Arabic", "Hebrew", "Hindi",
    "Chinese (Simplified)", "Chinese (Traditional)", "Japanese", "Korean",
]

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(32).hex())
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

APP_PASSWORD = os.environ.get("APP_PASSWORD", "translate123")

job_queues: dict[str, queue.Queue] = {}
job_outputs: dict[str, str] = {}


def emit(q: queue.Queue, **kwargs):
    q.put(kwargs)


def run_translation(job_id: str, input_path: str, target_language: str, dpi: int):
    q = job_queues[job_id]
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            emit(q, error="OPENAI_API_KEY is not configured on the server.")
            return

        output_path = str(OUTPUT_DIR / f"{job_id}.pdf")
        pages_dir   = str(UPLOAD_DIR / job_id / "pages")
        Path(pages_dir).mkdir(parents=True, exist_ok=True)

        page_renderer = PageRenderer(dpi=dpi)
        ocr           = OCRProcessor()
        eraser        = BackgroundEraser()
        translator    = Translator(api_key=api_key, target_language=target_language)
        text_renderer = TextRenderer(FONT_REGULAR, FONT_BOLD)
        assembler     = PDFAssembler()

        emit(q, step="render", message="Rendering document pages…", progress=5)
        pages = page_renderer.render(input_path, output_dir=pages_dir)
        total = len(pages)
        emit(q, step="render", message=f"Rendered {total} page(s) at {dpi} DPI.", progress=10)

        output_image_paths = []
        page_weight = max(1, 80 // total)

        for page in pages:
            idx  = page["page_index"]
            n    = idx + 1
            base = 10 + idx * page_weight

            emit(q, step="ocr",
                 message=f"Running OCR — page {n} of {total}…",
                 progress=base + int(page_weight * 0.15),
                 page=n, total=total)
            words = ocr.extract(page["image_path"])

            if not words:
                output_image_paths.append(page["image_path"])
                emit(q, step="ocr",
                     message=f"No text found on page {n}, skipping.",
                     progress=base + page_weight, page=n, total=total)
                continue

            emit(q, step="layout",
                 message=f"Analysing layout — page {n} of {total}…",
                 progress=base + int(page_weight * 0.30),
                 page=n, total=total)
            lines = build_lines(words)

            if not lines:
                output_image_paths.append(page["image_path"])
                continue

            emit(q, step="translate",
                 message=f"Translating {len(lines)} lines to {target_language} — page {n} of {total}…",
                 progress=base + int(page_weight * 0.55),
                 page=n, total=total)
            payload    = [{"id": i, "text": line["text"]} for i, line in enumerate(lines)]
            translated = translator.translate_blocks(payload)
            id_map     = {t["id"]: t["translated"] for t in translated}
            for i, line in enumerate(lines):
                line["translated"] = id_map.get(i, "")

            emit(q, step="draw",
                 message=f"Rendering translated text — page {n} of {total}…",
                 progress=base + int(page_weight * 0.80),
                 page=n, total=total)
            image    = Image.open(page["image_path"]).convert("RGB")
            bg_color = eraser.sample_background(image)
            image    = text_renderer.overlay_lines(image, lines, bg_color)

            out_path = page["image_path"].replace(".png", "_translated.png")
            image.save(out_path)
            output_image_paths.append(out_path)

        emit(q, step="assemble", message="Assembling final PDF…", progress=92)
        assembler.assemble(output_image_paths, output_path, dpi=dpi)

        job_outputs[job_id] = output_path
        emit(q, step="done", message="Translation complete!", progress=100, done=True)

    except Exception as exc:
        emit(q, error=str(exc))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/app")
def index():
    if not session.get("authenticated"):
        return redirect(url_for("login"))
    return render_template("index.html", languages=SUPPORTED_LANGUAGES)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form.get("password") == APP_PASSWORD:
            session["authenticated"] = True
            return redirect(url_for("index"))   # /app
        error = "Incorrect password. Please try again."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/translate", methods=["POST"])
def translate():
    if not session.get("authenticated"):
        return jsonify({"error": "Not authenticated"}), 401

    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No file selected."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type '{ext}'. Upload a PDF or image."}), 400

    language = request.form.get("language", "French")
    try:
        dpi = max(72, min(int(request.form.get("dpi", 150)), 400))
    except ValueError:
        dpi = 150

    job_id   = str(uuid.uuid4())
    job_dir  = UPLOAD_DIR / job_id / "pages"
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = str(UPLOAD_DIR / job_id / f"input{ext}")
    file.save(input_path)

    job_queues[job_id] = queue.Queue()
    threading.Thread(
        target=run_translation,
        args=(job_id, input_path, language, dpi),
        daemon=True,
    ).start()

    return jsonify({"job_id": job_id})


@app.route("/progress/<job_id>")
def progress(job_id):
    if not session.get("authenticated"):
        return Response("Not authenticated", status=401)

    q = job_queues.get(job_id)
    if q is None:
        return Response("Job not found", status=404)

    def generate():
        while True:
            try:
                event = q.get(timeout=30)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("done") or event.get("error"):
                    break
            except queue.Empty:
                yield 'data: {"heartbeat":true}\n\n'

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/download/<job_id>")
def download(job_id):
    if not session.get("authenticated"):
        return redirect(url_for("login"))
    output_path = job_outputs.get(job_id)
    if not output_path or not Path(output_path).exists():
        return "File not ready or not found.", 404
    return send_file(output_path, as_attachment=True, download_name="translated.pdf")


@app.errorhandler(413)
def file_too_large(_):
    return jsonify({"error": "File too large. Maximum size is 50 MB."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
