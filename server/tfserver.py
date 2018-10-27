import tensorflow as tf
from tensorpack.predict.base import OfflinePredictor
from tensorpack.predict.config import PredictConfig
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.tfutils.sessinit import ChainInit
from models import Net2
from data_load import Net2DataFlow, get_mfccs_and_spectrogram
from audio import spec2wav, inv_preemphasis, db2amp, denormalize_db
from hparam import hparam as hp
import numpy as np

class TensorflowEvaluator:
    def __init__(self):
        self.logdir1 = ('home/sergey/Documents/vc/train1/timit/checkpoint')
        self.logdir2 = ('home/sergey/Documents/vc/train2/arctic/checkpoint')
        # Load graph
        self.model = Net2()
        ckpt1 = tf.train.latest_checkpoint(self.logdir1)
        ckpt2 = tf.train.latest_checkpoint(self.logdir2)
        print(ckpt1)
        # session_inits = []
        # if ckpt2:
        #     session_inits.append(SaverRestore(ckpt2))
        # if ckpt1:
        #     session_inits.append(SaverRestore(ckpt1, ignore=['global_step']))
        # pred_conf = PredictConfig(
        #     model=self.model,
        #     input_names=self.get_eval_input_names(),
        #     output_names=self.get_eval_output_names(),
        #     session_init=ChainInit(session_inits))
        # self.predictor = OfflinePredictor(pred_conf)

    def predict(self, path_to_wav):
        x_mfcc, y_spec, mel = get_mfccs_and_spectrogram(path_to_wav)
        pred_spec, y_spec, ppgs = self.predictor(x_mfcc, y_spec, mel)
        pred_spec = denormalize_db(pred_spec, hp.default.max_db, hp.default.min_db)
        y_spec = denormalize_db(y_spec, hp.default.max_db, hp.default.min_db)
        pred_spec = db2amp(pred_spec)
        y_spec = db2amp(y_spec)
        pred_spec = np.power(pred_spec, hp.convert.emphasis_magnitude)
        y_spec = np.power(y_spec, hp.convert.emphasis_magnitude)
        # Spectrogram to waveform
        audio = np.array(
            map(lambda spec: spec2wav(spec.T, hp.default.n_fft, hp.default.win_length, hp.default.hop_length,
                                      hp.default.n_iter), pred_spec))
        y_audio = np.array(
            map(lambda spec: spec2wav(spec.T, hp.default.n_fft, hp.default.win_length, hp.default.hop_length,
                                      hp.default.n_iter), y_spec))



    def get_eval_input_names(self):
        return ['x_mfccs', 'y_spec']


    def get_eval_output_names(self):
        return ['net2/eval/summ_loss']

if __name__ == '__main__':
    evaluator = TensorflowEvaluator()
    evaluator.predict('/datasets/arctic/bdl/arctic_a0001.wav')