import os
from flask import Flask, request, render_template
from train_model import generate_caption

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("index.html", caption="❌ No file part in the request.")

    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", caption="⚠️ No file selected. Please upload an image.")

    filepath = "uploaded_image.jpg"
    file.save(filepath)
    caption = generate_caption(filepath)
    return render_template("index.html", caption=caption)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
