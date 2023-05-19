
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

# use an integer called user_id to grab a user
@app.route("/users/<int:user_id>/")
def greet_user(user_id):
    users = ["Bob", "Jane", "Adam"]
    try:
        return f"<h2>Hi {users[user_id]}</h2>"
    except IndexError:
        abort(404)
