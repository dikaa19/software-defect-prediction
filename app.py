from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

selected_features = ['LOC_BLANK', 'BRANCH_COUNT', 'CYCLOMATIC_COMPLEXITY', 'LOC_EXECUTABLE',
       'HALSTEAD_DIFFICULTY', 'HALSTEAD_ERROR_EST', 'HALSTEAD_LENGTH',
       'HALSTEAD_LEVEL', 'HALSTEAD_PROG_TIME', 'HALSTEAD_VOLUME',
       'NUM_OPERANDS', 'NUM_OPERATORS', 'NUM_UNIQUE_OPERANDS',
       'NUM_UNIQUE_OPERATORS', 'LOC_TOTAL']

# Load the model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return "No file uploaded", 400

    # Memproses file yang diupload
    if file.filename.endswith('.csv'):
        data = pd.read_csv(file)
    elif file.filename.endswith('.xlsx'):
        data = pd.read_excel(file)
    else:
        return "Unsupported file type", 400

    # Memastikan bahwa data input memiliki fitur yang diperlukan
    missing_features = [feature for feature in selected_features if feature not in data.columns]
    if missing_features:
        return f"Missing features: {', '.join(missing_features)}", 400

    # Menyaring hanya fitur yang relevan
    data_selected = data[selected_features]

    # Prediksi
    predictions = model.predict(data_selected)

    # Menyusun hasil dalam format DataFrame
    result = pd.DataFrame({
        "Row": range(1, len(predictions) + 1),
        "Defect Prediction": predictions
    })

    total_data = len(result)
    defect_count = result[result['Defect Prediction'] == 1].shape[0]
    defect_results = result[result['Defect Prediction'] == 1]

    if defect_results.empty:
        return "No defects found in the predictions."
    defect_results_html = defect_results.to_html(classes='table table-striped')
    defect_summary = f"Total Data: {total_data} | Defects: {defect_count} ({(defect_count/total_data)*100:.2f}% of total data)"
    return render_template('result.html', defect_summary=defect_summary, tables=[defect_results_html], titles=['Defect Predictions'])

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory('.', filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
