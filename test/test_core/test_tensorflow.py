from gempy.core.interp_methods.tf_2D import Tensorflow_kriging_2D
import tensorflow as tf


def test_basic_tf():
    dips_position = tf.constant([[1,0],
                                 [1,1]],
                                dtype = tf.float32)
    surface_position = tf.constant([[[0,0],[2,0]],
                                    [[0,2],[2,2]]],dtype = tf.float32)
    dips = tf.constant([[1,1],
                        [1,1]],dtype= tf.float32)

    test_model = Tensorflow_kriging_2D(dips_position,surface_position,dips)

    test_model.squared_euclidean_distance(test_model.layer1,test_model.layer1)



