#! /usr/bin/env python

from django.http import HttpResponse
import tensorflow as tf
import numpy as np
import os

from django.shortcuts import render
from sklearn.externals import joblib
from tensorflow.contrib import learn
import json

batch_size = 64
data_source= "all.pickle"
eval_train = False
dev_sample_percentage =  0.2
allow_soft_placement = True
log_device_placement = False


input_dir =   os.path.dirname(os.path.abspath(__file__))
# input_dir = r"C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\pycharm_project\mysite\intention"
le = joblib.load( os.path.join(input_dir,"model_cnn","model_label_encode.pickle") )

# X = 'Châm_cứu là gì ?'
#
# X = [X,]
vocab_path = os.path.join(input_dir, "model_cnn", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
# Evaluation
# ==================================================
checkpoint_dir = os.path.join(input_dir,"model_cnn","checkpoints")
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=allow_soft_placement,
      log_device_placement=log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)



clf = joblib.load(os.path.join(input_dir, "model_svm", "model.pickle"))
count_vectorizer = joblib.load(os.path.join(input_dir, "model_svm", "transform_feature.pickle"))

def predict_cnn(request):
    input = request.GET.get("sentence","")
    X = [input, ]
    x = np.array(list(vocab_processor.transform(X)))

    with sess.as_default():
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        scores = graph.get_operation_by_name("output/scores").outputs[0][0]
        softmax_scores = tf.nn.softmax(scores)
        # batch_predictions = sess.run(predictions, {input_x: x, dropout_keep_prob: 1.0})
        prediction_eval, score_predict = sess.run([predictions, softmax_scores], {input_x: x, dropout_keep_prob: 1.0})
        score_predict = max(score_predict)

    all_predictions = le.inverse_transform(np.array(prediction_eval, np.int))[0]
    print(all_predictions)
    print(score_predict)
    # result = {"predict":all_predictions,"prob":"{0:.2f}".format(score_predict.item())}


    result = {"predict":all_predictions}
    return HttpResponse(json.dumps(result))


def predict_svm(request):
    input = request.GET.get("sentence","")
    #input = "nam"
    X = [input, ]
    X_feature_test = count_vectorizer.transform(X)
    y_pred = clf.predict(X_feature_test)[0]
    result = {"predict":y_pred}
    return HttpResponse(json.dumps(result))



def index_cnn(request):
    return render(request, 'index_cnn.html')
    # return HttpResponse("ok")
def index_svm(request):
    return render(request, 'index_svm.html')