from flask import Flask
from .routes import bp as main_bp
from app.cvae import cVAE
from app.config import PTH_PATH, OUTPUT_DIM, INPUT_DIM, X_SCALER, Y_SCALER, DEVICE
from tortreinador.utils.View import init_weights
import joblib
import torch

# use flask --app app:create_app --debug run
def create_app():
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY='dev',
    )

    cvae_model = cVAE(i_dim=OUTPUT_DIM, z_dim=int(OUTPUT_DIM * 7), num_hidden=1024,
                      c_dim=INPUT_DIM, o_dim=OUTPUT_DIM)
    init_weights(cvae_model)

    if DEVICE.type == 'cpu':
        cvae_model.load_state_dict(torch.load(PTH_PATH, weights_only=True, map_location=torch.device('cpu')))
        print("Loaded cVAE in CPU")

    else:
        cvae_model.load_state_dict(torch.load(PTH_PATH, weights_only=True))
        print("Loaded cVAE in GPU")

    cvae_model.to(DEVICE)
    cvae_model.eval()

    s_x = joblib.load(X_SCALER)
    s_y = joblib.load(Y_SCALER)

    app.config['MODEL'] = cvae_model
    app.config['x_scaler'] = s_x
    app.config['y_scaler'] = s_y

    app.register_blueprint(main_bp, url_prefix='/api')

    @app.before_request
    def before_request():
        pass

    @app.after_request
    def after_request(response):
        return response

    return app
