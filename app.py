import os
import numpy as np
import io
import base64
from flask import Flask, request, render_template, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# 업로드 폴더 설정
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 모델 로드
model = load_model('model/animal_face_model.h5')

# 클래스 이름 정의
class_names = ['고양이상', '강아지상', '여우상', '토끼상']

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 웹캠 이미지가 있을 경우
    if 'webcam_image' in request.form:
        data_url = request.form['webcam_image']
        header, encoded = data_url.split(",", 1)
        img_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(img_data)).resize((224, 224)).convert('RGB')

        # 파일로 저장하여 처리
        filename = 'webcam_capture.png'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(file_path)

        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

    # 파일 업로드 처리
    else:
        if 'file' not in request.files:
            return render_template('error.html', error='No file part')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('error.html', error='No selected file')
        
        if file:
            # 파일 저장
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # 이미지 처리
            image = Image.open(file_path).resize((224, 224)).convert('RGB')
            img_array = img_to_array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

    # 예측 수행
    predictions = model.predict(img_array)[0]
    confidence = predictions / np.sum(predictions)
    confidence_list = confidence.tolist()

    # 가장 높은 확률의 인덱스 찾기
    predicted_class_index = np.argmax(confidence)

    return render_template('result.html', 
                           class_names=class_names, 
                           confidence=confidence_list,
                           predicted_class=class_names[predicted_class_index],
                           image_file=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
