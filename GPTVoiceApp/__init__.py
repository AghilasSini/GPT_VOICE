from flask import Flask
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(16)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False

from GPTVoiceApp import views
