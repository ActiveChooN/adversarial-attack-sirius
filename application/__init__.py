from flask import Flask
from flask_bootstrap import Bootstrap
from flask_config import config

app = Flask(__name__)


def create_app(config_name):
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    Bootstrap(app)

    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app
