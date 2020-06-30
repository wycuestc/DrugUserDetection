from keras.layers import Dense, LSTM, Dropout, Embedding, SpatialDropout1D, Bidirectional, concatenate, InputSpec

from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras import backend as K

from sklearn.preprocessing import LabelEncoder

class AttentionWeightedAverage(Layer):

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(**kwargs)

    def build(self, input_shape):  # input_shape: (None, 5, 150)
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.w = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_w'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.w]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, h, mask=None):
        h_shape = K.shape(h)  # h:(None, 5, 150)  w:(150, 1)
        d_w, T = h_shape[0], h_shape[1]
        logits = K.dot(h, self.w)  # w^T h (it is actually calculated by (h^T * w)^T)        #(None, 5, 1)
        logits = K.reshape(logits, (d_w, T))  # transpose + convert to two dimensions: (1, 5)
        alpha = K.exp(logits - K.max(logits, axis=-1, keepdims=True))  # exp

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            alpha = alpha * mask
        alpha = alpha / K.sum(alpha, axis=1, keepdims=True)  # softmax # (1, 5)
        r = K.sum(h * K.expand_dims(alpha),
                  axis=1)  # r = h*alpha^T  # element product (None, 5, 150) * (1, 5, 1) # (None, 150)
        h_star = K.tanh(r)  # h^* = tanh(r) #(None, 150)
        if self.return_attention:
            return [h_star, alpha]
        return h_star

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None