## basically entirely written from a repo in GitHub called https://github.com/rhnvrm/galaxy-image-classifier-tensorflow.git
#which was able to show me the correct syntax and apply it to my flower purposes
#Let's hope all I need to pass in are my own labels and then we're good, as of 5/3 I have not changed anything

#need to make sure that you have Docker locally on you computer for this to work... took me a while to configure that, sorry for making this code a little eextra difficult to run
#changed so instead of automatically taking sys.argv[1] instead I'm taking an input to make it interactive

import tensorflow as tf, sys
print("Include the path to the image you'd like to identify:")

image_path = input()

# Read in the image_data
image_data = tf.io.gfile.GFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.io.gfile.GFile("tf_files/flower_labels.txt")]

# Unpersists graph from file
with tf.compat.v1.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.compat.v1.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
