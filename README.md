# cVAE API

## Overview
The cVAE API is a Flask-based service that exposes a conditional variational autoencoder (cVAE) trained to predict the interior structure of rocky exoplanets. The application loads a pretrained PyTorch model together with feature scalers and offers JSON as well as file-based inference endpoints so you can request probabilistic distributions for interior structure parameters from bulk planetary measurements.

## Features
- Loads a pretrained cVAE model and the associated feature scalers at startup for deterministic inference results.
- Supports scalar JSON inputs, batched JSON inputs, and file uploads (NumPy, CSV, Excel, and Parquet) for prediction requests.
- Automatically handles missing `Times` values by falling back to a default of 10 predictive samples per input row.
- Returns distributions for eight interior structure parameters, including weight and composition-related factors, in SI-friendly units.

## Project structure
```
.
├── app/
│   ├── __init__.py        # Flask application factory and model initialisation
│   ├── routes.py          # REST API endpoints and inference helpers
│   ├── config.py          # Shared configuration constants
│   └── cvae.py            # Model architecture definitions
├── static/                # Pretrained weights and scaler artefacts
├── requirements.txt       # Python runtime dependencies
└── gunicorn_conf.py       # Production Gunicorn configuration
```

## Requirements
- Python 3.10+
- Dependencies listed in `requirements.txt` (Flask, PyTorch, pandas, etc.).
- Pretrained assets placed in `static/best_model.pth`, `static/Xscaler.save`, and `static/yscaler.save`.
- At least 1 GB of available memory to load the pretrained model and scalers.

## Local development
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Export the Flask application entry point and start the development server (binding to port `8000`):
   ```bash
   export FLASK_APP=app:create_app
   flask run --debug --port 8000
   ```
   The API will be available at `http://127.0.0.1:8000/api/`.

To run the service with Gunicorn (recommended for production):
```bash
gunicorn --config gunicorn_conf.py 'app:create_app()'
```

## API reference
All endpoints are namespaced under `/api`.

### `GET /api/hello`
Health-check endpoint that returns a plain text greeting.

```bash
curl http://127.0.0.1:8000/api/hello
```

### `POST /api/single_prediction`
Accepts JSON containing scalar or vector inputs for the four required features. Each feature should be provided as a list to allow batch predictions. Optional `Times` controls the number of samples drawn for each input (default 10). Example:
```json
{
  "Mass": [1.0],
  "Radius": [1.0],
  "Fe/Mg": [0.5],
  "Si/Mg": [1.0],
  "Times": 25
}
```
The response contains the received payload and a `Prediction_distribution` map keyed by the batch index.

```bash
curl -X POST http://127.0.0.1:8000/api/single_prediction \
  -H "Content-Type: application/json" \
  -d '{
        "Mass": [1.0],
        "Radius": [1.0],
        "Fe/Mg": [0.5],
        "Si/Mg": [1.0],
        "Times": 25
      }'
```

### `POST /api/multi_prediction`
Identical payload contract as `single_prediction`. This endpoint is intended for submitting multiple planetary cases at once.

### `POST /api/file_prediction`
Accepts multipart form-data with:
- `file`: uploaded `.npy`, `.csv`, `.xlsx`, or `.parquet` file containing the four feature columns in the order `[Mass, Radius, Fe/Mg, Si/Mg]`.
- Optional `Times` form field (defaults to 10).

Returns a JSON object that contains the predicted distributions for each input row. Mass and radius derived quantities in the output are rescaled to original units before being returned.

## Model outputs
Predictions include distributions for the following parameters:
- `WRF` – water radial fraction
- `MRF` – mantle radial fraction
- `CRF` – core radial fraction
- `WMF` – water mass fraction
- `CMF` – core mass fraction
- `P_CMB (TPa)` – core-mantle boundary pressure (terapascal)
- `T_CMB (10^3K)` – core-mantle boundary temperature (thousands of Kelvin)
- `K2` – tidal Love number

## Contributing
Pull requests are welcome. Please ensure code is formatted and that new features include suitable documentation updates.

## License
Specify the project license here if available.
