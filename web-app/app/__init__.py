from flask import Flask
import os

app=Flask(__name__,static_folder='static')
app.config['SECRET_KEY']='blablabla'

APP_ROOT=os.path.dirname(os.path.abspath(__file__))

from app import routes