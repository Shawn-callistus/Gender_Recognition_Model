from flask import Flask,render_template,Response
import cv2
import numpy as np
app=Flask(__name__)
camera=cv2.VideoCapture(0)

face=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def generate_frames():
     while True:
        success,frame=camera.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face.detectMultiScale(gray,1.1,3)
        for (x,y,w,h) in faces:
         endx=x+w
         endy=y+h
         cv2.rectangle(frame,(x,y),(endx,endy),(255,0,0),2)
        
         ret,buffer=cv2.imencode(".jpg",frame)
         frame=buffer.tobytes()
         yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




@app.route("/")
def index():
    return render_template("index.html")
@app.route("/video")
def video():
     return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(host="localhost",port=5000,debug=True)