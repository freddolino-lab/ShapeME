
from markupsafe import escape
from flask import Flask,abort

app = Flask(__name__)

@app.route("/")
@app.route("/index/")
def hello():
    return "<h1>Hello, World!</h1>"

@app.route("/about/")
def about():
    return "<h3>This is a Flask web application.</h3>"

# note <word> in the route allows passing of word as var to capitalize fn
@app.route("/capitalize/<word>/")
def capitalize(word):
    # use escape here to avoid Cross Site Scripting attacks
    return f"<h1>{escape(word.capitalize())}</h1>"

# now we only accept integers
@app.route("/add/<int:n1>/<int:n2>/")
def add(n1, n2):
    return f"<h1>{n1 + n2}</h1>"
