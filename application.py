from flask import Flask, render_template, redirect, request
import services.service as service
import os, shutil
import glob

application = Flask(__name__, static_folder='static', template_folder='templates')

UPLOAD_FOLDER = os.path.join('static', 'uploads')
application.secret_key = "secret key"
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
application.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
@application.route('/')
def home():
    return render_template('index_v2.html', href='/ndcv', is_have = True)

@application.route('/ndcv', methods=['GET'])
def ndcv():
    return render_template('index_v2.html', href='/ndcv', is_have = True, is_image = False)

@application.route('/ndcv', methods=['POST'])
def ndcv_post():
    
    if not os.path.exists(UPLOAD_FOLDER): # Create Directory for the uploaded static
        os.mkdir(UPLOAD_FOLDER)
    
    files = glob.glob(UPLOAD_FOLDER+"/*")
    for f in files:
        os.remove(f)
    
    _img = request.files['file-uploaded']
    filename = _img.filename
    file_extension = filename.split(".")[-1]
    _img.save(os.path.join(UPLOAD_FOLDER, '1.' + file_extension))
    data_predict = service.NDCV()
    print('type of predict',type(data_predict))
    img = os.path.join(UPLOAD_FOLDER, '1.' + file_extension)
    return render_template('index_v2.html', href='/ndcv', is_have = True, data_predict = data_predict, is_image = True, img = img)

@application.route('/ndgn', methods=['GET'])
def ndgn():
    return render_template('index_v2.html',href='/ndgn', is_have = False)
@application.route('/ndkm', methods=['GET'])
def ndkm():
    return render_template('index_v2.html',href='/ndkm', is_have = False)
@application.route('/ndvt', methods=['GET'])
def ndvt():
    return render_template('index_v2.html',href='/ndvt', is_have = False)
if __name__ == "__main__":
    application.debug = True
    application.run(host='0.0.0.0', port=5000)
    application.run()
