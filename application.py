from flask import Flask, render_template, redirect
import services.service as service

# from services import service

UPLOAD_FOLDER = 'static/uploads/'

application = Flask(__name__)
application.secret_key = "secret key"
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
application.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@application.route('/')
def home():
    return render_template('index.html', href='/ndcv')

@application.route('/ndcv', methods=['GET'])
def ndcv():
    return render_template('index.html', href='/ndcv')

@application.route('/ndcv', methods=['POST'])
def ndcv_post():
    data = "post method"
    predict = service.NNCV()
    return render_template('index.html', href='/ndcv', data=data, predict = predict, len=len(predict))

if __name__ == "__main__":
    application.debug = True
    application.run()
