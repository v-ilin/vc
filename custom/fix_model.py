import tensorflow as tf
import dual_net

save_file = '../epoch_12_step_6924.data-00000-of-00001'
dest_file = '../epoch_12_step_6924.data-00000-of-00001-upgrade'
features, labels = dual_net.get_inference_input()
dual_net.model_fn(features, labels, tf.estimator.ModeKeys.PREDICT,
                  dual_net.get_default_hyperparams())
sess = tf.Session()

# retrieve the global step as a python value
ckpt = tf.train.load_checkpoint(save_file)
global_step_value = ckpt.get_tensor('global_step')


# restore all saved weights, except global_step
from tensorflow.python.framework import meta_graph
meta_graph_def = meta_graph.read_meta_graph_file(save_file  '.meta')
stored_var_names = set([n.name
                        for n in meta_graph_def.graph_def.node
                        if n.op == 'VariableV2'])
print(stored_var_names)
stored_var_names.remove('global_step')
var_list = [v for v in tf.global_variables()
            if v.op.name in stored_var_names]
tf.train.Saver(var_list=var_list).restore(sess, save_file)

# manually set the global step
global_step_tensor = tf.train.get_or_create_global_step()
assign_op = tf.assign(global_step_tensor, global_step_value)
sess.run(assign_op)

# export a new savedmodel that has the right global step type
tf.train.Saver().save(sess, dest_file)