from flask import Flask, render_template, flash, url_for, redirect, request
from werkzeug.utils import secure_filename
import os
from app import app



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
   if request.method == 'POST':
      f = request.files['file']
      f.save(os.path.join(app.config['UPLOAD_FOLDER']), secure_filename(f.filename))
      return 'file uploaded successfully'



