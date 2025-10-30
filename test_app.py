import os
from flask import Flask, render_template

# Create a simple Flask app
app = Flask(__name__)

# Set the template folder explicitly
app.template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

@app.route('/')
def test():
    print(f"Template folder: {app.template_folder}")
    print(f"Template exists: {os.path.exists(os.path.join(app.template_folder, 'index.html'))}")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
