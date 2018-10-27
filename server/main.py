import matplotlib
matplotlib.use('Agg')
from flask import Flask, request, send_file
from flask.json import jsonify
import os
import io
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "audios"
ALLOWED_EXTENSIONS = set(['mp3', 'wav'])
#graph_params, sess = init_tf()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# def setup_graph():
#     graph_params = {}-
#     graph_params['graph'] = tf.Graph()
#     with graph_params['graph'].as_default():
#         model_params = model.params()
#         graph_params['target_image'] = tf.placeholder(
#             tf.float32,
#             shape=(1, common.CNN_IN_HEIGHT, common.CNN_IN_WIDTH,
#                    common.CNN_IN_CH))
#         logits = model.cnn(
#             graph_params['target_image'], model_params, keep_prob=1.0)
#         graph_params['pred'] = tf.nn.softmax(logits)
#         graph_params['saver'] = tf.train.Saver()
#     return graph_params


# def init_tf():
#     # Setup computation graph
#     graph_params = setup_graph()

#     # Model initialize
#     sess = tf.Session(graph=graph_params['graph'])
#     tf.global_variables_initializer()
#     if os.path.exists('models'):
#         save_path = os.path.join('models', 'deep_logo_model')
#         graph_params['saver'].restore(sess, save_path)
#         print('Model restored')
#     else:
#         print('Initialized')
#     return graph_params, sess


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def bad_request(reason):
    response = jsonify({"error": reason})
    response.status_code = 400
    return response


@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return bad_request("empty body")
        file = request.files['file']
        if file.filename == '':
            return bad_request("empty filename")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # global graph_params, sess
            # detect_logo.main(
            #     filename, graph_params=graph_params, sess=sess)
            # return send_file(io.BytesIO(img.read()), mimetype='image/jpg')
            return "hello"  # send_file(os.path.join("results", filename))
    except Exception as e:
        bad_request(e)


@app.route('/')
def hello_world():
    return ('hello world!')


if __name__ == '__main__':

    app.run()
