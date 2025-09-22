from modules.AI_models_utils import BaseModel


class TensorFlowMLPReg(BaseModel):
    def __init__(self, input_size=3, output_size=4, batch_size=200):
        super(TensorFlowMLPReg, self).__init__(input_size)
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        self.batch_size = batch_size

        self.tf = tf
        self.keras = keras
        self.layers = layers

        self.model = keras.Sequential([
            layers.InputLayer(input_shape=(input_size,)),
            layers.Dense(128, activation='sigmoid'),
            layers.Dense(128, activation='sigmoid'),
            layers.Dense(output_size)  # No activation on output for regression
        ])
        self.model.compile(optimizer='adam', loss='mse')
    
    def predict(self, X):
        return self.model.predict(X)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=20000, batch_size=self.batch_size, verbose=0)