# app.py
from flask import Flask, render_template, request, redirect, url_for, Response
from camera import generate_frames
from database import add_score, get_rankings

app = Flask(__name__)
user_name = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    global user_name
    user_name = request.form['name']
    return redirect(url_for('video'))

@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/result', methods=['POST'])
def result():
    global user_name
    accuracy = float(request.form['accuracy'])
    add_score(user_name, accuracy)
    return redirect(url_for('rankings'))

@app.route('/rankings')
def rankings():
    scores = get_rankings()
    return render_template('rankings.html', scores=scores)

if __name__ == '__main__':
    app.run(debug=True)
