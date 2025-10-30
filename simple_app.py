import os
from flask import Flask, render_template

app = Flask(__name__, template_folder=os.path.dirname(os.path.abspath(__file__)))

@app.route('/')
def home():
    return render_template('simple_index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5002)
