import numpy as np
from flask import Blueprint, request, jsonify, current_app
from flask.typing import ResponseReturnValue
from app.config import INPUT_PARAM, DEVICE, OUTPUT_PARAM
import pandas as pd
import io

bp = Blueprint('main', __name__)

def load_model_scaler():
    model = current_app.config['MODEL']
    x_scaler = current_app.config['x_scaler']
    y_scaler = current_app.config['y_scaler']

    return model, x_scaler, y_scaler

def inference(data, is_file=False):

    model, x_scaler, y_scaler = load_model_scaler()

    # If not receive the Times variable, set as default
    if "Times" not in data.keys():
        data['Times'] = 10

    input_data = []

    if not is_file:
        for p in INPUT_PARAM:
            input_data.append(data[p])

        input_data = np.vstack(input_data)
        input_data = input_data.transpose(1, 0)

    else:
        input_data = data['input']

    data_length = input_data.shape[0]
    x = x_scaler.transform(input_data)
    # input_data = torch.from_numpy(input_data).to(DEVICE)

    conditions, pre_distribution = model.pred_distribution_inference(x, data['Times'], DEVICE)
    pre_distribution = y_scaler.inverse_transform(pre_distribution)
    pre_distribution[:, 5] = pre_distribution[:, 5] / 1000
    pre_distribution[:, 6] = pre_distribution[:, 6] / 1000

    pre_distribution_split = np.array_split(pre_distribution, data_length, axis=0)
    out = {
        i: {} for i in range(data_length)
    }
    for i in range(data_length):

        for idx, out_p in enumerate(OUTPUT_PARAM):
            out[i][out_p] = pre_distribution_split[i][:, idx].tolist()

    return out

@bp.get('/hello')
def hello() -> ResponseReturnValue:
    return 'Hello World!'

@bp.post('/single_prediction')
def single_prediction() -> ResponseReturnValue:
    data = request.get_json()
    out = inference(data)

    return {'Receive': data, "Prediction_distribution": out}

@bp.post('/multi_prediction')
def multi_prediction() -> ResponseReturnValue:
    data = request.get_json()
    out = inference(data)
    return {"Receive": data, "Prediction_distribution": out}

@bp.post('/file_prediction')
def file_prediction() -> ResponseReturnValue:
    data = {}

    if 'file' not in request.files:
        return jsonify({'error': 'No file sent'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file.filename.endswith('npy'):
        data['input'] = np.load(io.BytesIO(file.read()))

    elif file.filename.endswith('csv'):
        data['input'] = pd.read_csv(io.BytesIO(file.read())).to_numpy()

    elif file.filename.endswith('xlsx'):
        data['input'] = pd.read_excel(io.BytesIO(file.read())).to_numpy()

    elif file.filename.endswith('parquet'):
        data['input'] = pd.read_parquet(io.BytesIO(file.read())).to_numpy()

    else:
        return jsonify({"error": "Unsupported file type"}), 400

    if 'Times' not in request.form:
        times = 10

    else:
        times = int(request.form['Times'])

    data['Times'] = times

    out = inference(data, is_file=True)
    return {"Prediction_distribution": out}