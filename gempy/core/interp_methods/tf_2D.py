import tensorflow as tf

"""

This is a module creating scaler field through tensorflow instead of Theano

"""
def constant(val,dtype = tf.float32):
    """

    Args:
        val: python number or tuple

    Returns:
        tensorflow constant object

    """
    return tf.constant(val, dtype = dtype)

class Tensorflow_kriging_2D(object):

    def __init__(self, dips_position, layers_position, dips):
        """
        Kriging using Tensorflow > 2.0

        Args:
            dips_position: x,y
            layers_position:
            dips:
            dtype: tensorflow dtype, default = tf.float32

        """
        self.layers_position = layers_position
        self.dips_position = dips_position
        self.dips = dips
        self.number_of_layers = len(self.layers_position)

        ### let's only have two layer first
        self.layer1 = layers_position[0]
        self.layer2 = layers_position[1]

        self.dips_position_tile = tf.tile(self.dips_position, [2, 1])
        self.a_T = 5
        self.c_o_T = self.a_T ** 2 / 14 / 3

        self.number_of_points_per_surface = constant([self.layer1.shape[0], self.layer2.shape[0]])


    def squared_euclidean_distance(self,x_1,x_2):
        sqd = tf.sqrt(tf.reshape(tf.reduce_sum(x_1**2,1),shape =(x_1.shape[0],1))+\
        tf.reshape(tf.reduce_sum(x_2**2,1),shape =(1,x_2.shape[0]))-\
        2*tf.tensordot(x_1,tf.transpose(x_2),1))
        return sqd
