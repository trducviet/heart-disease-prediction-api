from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'models'

# Load model và pipeline đã lưu
model = joblib.load(MODEL_DIR / 'logistic_model.pkl')
scaler = joblib.load(MODEL_DIR / 'scaler.pkl')
selector = joblib.load(MODEL_DIR / 'selector.pkl')
le_dict = joblib.load(MODEL_DIR / 'label_encoders.pkl')

# Danh sách tất cả các cột đầu vào (theo dữ liệu gốc)
all_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
               'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Trang chủ - hiển thị hướng dẫn sử dụng API
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Chào mừng đến với API Dự đoán Bệnh Tim',
        'endpoints': {
            '/predict': {
                'method': 'POST',
                'description': 'Dự đoán bệnh tim dựa trên dữ liệu đầu vào',
                'required_fields': all_columns,
                'example_input': {
                    'input': {
                        'Age': 40,
                        'Sex': 'M',
                        'ChestPainType': 'ATA',
                        'RestingBP': 140,
                        'Cholesterol': 289,
                        'FastingBS': 0,
                        'RestingECG': 'Normal',
                        'MaxHR': 172,
                        'ExerciseAngina': 'N',
                        'Oldpeak': 0,
                        'ST_Slope': 'Up'
                    }
                }
            }
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data_json = request.get_json()
        if not data_json or 'input' not in data_json:
            return jsonify({'error': 'Thiếu trường "input" trong dữ liệu gửi lên'}), 400
        
        input_data = data_json['input']

        # Kiểm tra đủ các cột
        missing_cols = [col for col in all_columns if col not in input_data]
        if missing_cols:
            return jsonify({'error': f'Thiếu các trường dữ liệu: {missing_cols}'}), 400

        # Chuyển dữ liệu input thành DataFrame 1 dòng
        df = pd.DataFrame([input_data], columns=all_columns)

        # Chuyển kiểu dữ liệu đúng
        for col in ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Mã hóa categorical bằng LabelEncoder đã lưu
        for col in categorical_cols:
            le = le_dict.get(col)
            if le is None:
                return jsonify({'error': f'Không tìm thấy LabelEncoder cho cột {col}'}), 500
            # Nếu giá trị input không trong lớp của encoder, trả lỗi
            if df.at[0, col] not in le.classes_:
                return jsonify({
                    'error': f'Giá trị "{df.at[0, col]}" ở cột {col} không hợp lệ',
                    'allowed_values': le.classes_.tolist()
                }), 400
            df[col] = le.transform([df.at[0, col]])

        # Xử lý missing value (nếu có)
        if df.isnull().any().any():
            return jsonify({'error': 'Dữ liệu có giá trị không hợp lệ hoặc thiếu'}), 400

        # Chuẩn hóa dữ liệu
        X_scaled = scaler.transform(df)

        # Chọn các đặc trưng quan trọng
        X_selected = selector.transform(X_scaled)

        # Dự đoán
        prediction = model.predict(X_selected)[0]
        prediction_proba = model.predict_proba(X_selected)[0]
        prediction_label = 'Có bệnh tim' if prediction == 1 else 'Không có bệnh tim'

        return jsonify({
            'prediction': prediction_label,
            'probability': {
                'Không có bệnh tim': float(prediction_proba[0]),
                'Có bệnh tim': float(prediction_proba[1])
            }
        })

    except Exception as e:
        return jsonify({'error': f'Lỗi server: {str(e)}'}), 500

# Xử lý lỗi 404 - Không tìm thấy trang
@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'error': 'Không tìm thấy địa chỉ này',
        'message': 'Vui lòng truy cập "/" để xem hướng dẫn sử dụng API'
    }), 404

# Xử lý lỗi 405 - Phương thức không được phép
@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({
        'error': 'Phương thức không được phép',
        'message': 'Vui lòng kiểm tra phương thức HTTP (GET/POST) của request'
    }), 405

if __name__ == '__main__':
    app.run(debug=False, port=8080)
