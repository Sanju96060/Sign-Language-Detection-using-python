from flask import Flask, render_template, jsonify
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/detection')
def detection():
    # This is where you trigger the detection script.
    subprocess.Popen(['python', 'test.py'])
    return render_template('detection.html')

@app.route('/stop_detection')
def stop_detection():
    # Here you would stop the detection process
    # In this example, it's not implemented.
    return jsonify({'status': 'stopped'})

if __name__ == '__main__':
    app.run(debug=True)

