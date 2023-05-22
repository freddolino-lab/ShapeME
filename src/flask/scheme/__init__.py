import os

from flask import Flask

def create_app(test_config=None):
    # create and configure the app
    app = Flask(
        # name of current python module
        __name__,
        # True tells app that conf files are relative to the "instance folder"
        instance_relative_config=True,
    )
    app.config.from_mapping(
        # dev good for development, should be random value when deploying
        SECRET_KEY = "dev",
        # where the sqlite database will be written
        DATABASE = os.path.join(app.instance_path, "flaskr.sqlite"),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        # apparently this can be used to set a real SECRET_KEY
        app.config.from_pyfile("config.py", silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route("/hello")
    def hello():
        return "Hello, World!"

    return app
