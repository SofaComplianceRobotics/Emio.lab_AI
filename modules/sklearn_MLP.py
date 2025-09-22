from modules.AI_models_utils import BaseModel


class SklearnMLPReg(BaseModel):
    def __init__(self, input_size=3, output_size=4, batch_size=200, model_file=None):
        super(SklearnMLPReg, self).__init__(input_size)
        from sklearn.neural_network import MLPRegressor

        self.batch_size = batch_size
        self.model = MLPRegressor(hidden_layer_sizes=(128, 128), activation='logistic', solver='adam',
                                  max_iter=20000, batch_size=self.batch_size, random_state=1)
        if model_file:
            self.load(model_file)
        
    def save(self, file_path):
        import joblib
        joblib.dump(self.model, file_path)

    def load(self, file_path):
        import joblib
        self.model = joblib.load(file_path)

    def predict(self, X):
        return self.model.predict(X)
    
    def train(self, X_train, y_train):
        print(f"[SklearnMLPReg] Training model with {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        print("[SklearnMLPReg] Training completed.")

    def score(self, X_test, y_test) -> float:
        r2 = self.model.score(X_test, y_test)
        return r2

