import os
import base64
from yolo_classifier import YOLOClassifier
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

labels = ["NORMAL", "PNEUMONIA"]

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_path = 'data/weights/yolo_96_acc.pt' 
classifier = YOLOClassifier(model_path)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'files[]' not in request.files:
            return redirect(request.url)
        files = request.files.getlist('files[]')
        results = []

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                predicted_label = classifier.predict(file_path)

                with open(file_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')

                results.append({
                    "image_data": img_data,
                    "predicted_label": labels[predicted_label],
                    "filename": filename
                })

        return render_template('results.html', results=results)

    return render_template('index.html')

@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return render_template('shutdown.html')

def shutdown_server():
    """Функция завершения работы Flask-сервера."""
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        print("Not running with the Werkzeug Server. Forcing exit.")
        os._exit(0)  # Принудительное завершение процесса
    func()

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
