from flask import Flask, request, render_template, redirect, url_for
import os, shutil, torch
from pathlib import Path

app = Flask(__name__)

# ───────────── ディレクトリ設定 ─────────────
STATIC_BASE  = Path("static")
UPLOAD_DIR   = STATIC_BASE / "uploads"    # 元画像
PREDICT_DIR  = STATIC_BASE / "predict"    # 推論結果
for d in (UPLOAD_DIR, PREDICT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ───────────── YOLOv5 モデル ─────────────
model = torch.hub.load("yolov5", "yolov5s", source="local")

@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        f = request.files.get("file")
        if f and f.filename:

            # 🔥 static フォルダを完全リセット ────────────────
            if STATIC_BASE.exists():
                shutil.rmtree(STATIC_BASE)
            for d in (UPLOAD_DIR, PREDICT_DIR):       # 空で再生成
                d.mkdir(parents=True, exist_ok=True)

            # 1⃣ アップロード画像保存
            src_path = UPLOAD_DIR / f.filename
            f.save(src_path)

            # 2⃣ 推論 → PREDICT_DIR に保存
            results = model(str(src_path))
            results.save(save_dir=str(PREDICT_DIR))

            # 3⃣ 保存された実ファイル名を取得
            pred_name = Path(results.files[0]).name    # 例: image0.jpg
            return redirect(url_for("result", filename=pred_name))

    return render_template("index.html")


@app.route("/result/<filename>")
def result(filename):
    return render_template("result.html", filename=filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

