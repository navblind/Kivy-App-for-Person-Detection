import os,time,cv2, math
import tensorflow as tf
import numpy as np

from utils import testUtils as utils
from utils import helpers
from builders import model_builder


class Segment():
    class_names_list = None
    label_values = None
    num_classes = 0
    config = None
    sess = None
    net_input = None
    net_output = None
    network = None
    saver = None
    width = ""
    height = ""
    checkpoints = None
    model = ""
    checkpoints = ""

    def __init__(self, modelName, width, height, checkpoints):
        
        self.model = modelName
        self.width = width
        self.height = height
        self.checkpoints = checkpoints

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        self.class_names_list, self.label_values = helpers.get_label_info("class_dict.csv")
        self.num_classes = len(self.label_values)
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        self.sess=tf.Session(config=self.config)
        self.net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
        self.net_output = tf.placeholder(tf.float32,shape=[None,None,None,self.num_classes])
        
        self.network, _ = model_builder.build_model(modelName, net_input=self.net_input,
                                        num_classes=self.num_classes,
                                        crop_width=width,
                                        crop_height=height,
                                        is_training=False)
        self.sess.run(tf.global_variables_initializer())
        print('Loading model checkpoint weights') 
        self.saver=tf.train.Saver(max_to_keep=1000)
        self.saver.restore(self.sess, checkpoints)
        
        print("Setup done!")

    def seg(self, img):
        print()
        loaded_image = utils.load_image(img)
        resized_image =cv2.resize(img, (int(self.width),int(self.height)))
        input_image = np.expand_dims(np.float32(resized_image[:int(self.height), :int(self.width)]),axis=0)/255.0
        output_image = self.sess.run(self.network,feed_dict={self.net_input:input_image})

        output_image = np.array(output_image[0,:,:,:])
        output_image = helpers.reverse_one_hot(output_image)

        out_vis_image = helpers.colour_code_segmentation(output_image, self.label_values)
        
        return cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR)



if __name__=="__main__":
    obj = Segment("MobileUNet", "512", "512", "checkpoints/model.ckpt")
    img = cv2.imread("test.jpg")

    im = obj.seg(img)

    cv2.imshow("im", im)
    cv2.waitKey()
