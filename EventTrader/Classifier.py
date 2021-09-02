import Feature_Generator
from Auction_model.Config import Auction_params as PARAMS
from Logging import logger
import numpy as np
import math
from datetime import timedelta
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from Hyperoptimization import HPOpt
from hyperopt import space_eval
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore')
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


class BinaryClassifier:

    def __init__(self, dates, prices, x_predict_end, direction):
        self.HP_opt = PARAMS['classifier_params']['HP_opt']
        self.train_test_percent = PARAMS['classifier_params']['Train_test']
        self.features_window = PARAMS['classifier_params']['Features_window']
        self.feature_sel = PARAMS['classifier_params']['Feature_sel']
        self.folds = PARAMS['classifier_params']['Folds']
        self.classifier = PARAMS['classifier_params']['Algorithm']
        self.prices = prices
        self.direction = direction
        self.x_predict_end = x_predict_end
        self.x_predict_start = None
        self.agent_dates = self.get_x_dates(dates)
        self.log_returns = prices.dropna()
        self.asset = self.log_returns.name
        self.features = None
        self.X = None
        self.X_new = None
        self.y = None
        self.y_train = None
        self.y_test = None
        self.X_train = None
        self.X_test = None
        self.model = None
        self.feature_model = None
        self.training_score = None
        self.roc_auc = None

    def get_x_dates(self, dates):
        df = dates.rename(columns={'first': 'y_start', 'last': 'y_end'})
        xl = []
        for index, row in df.iterrows():
            xl.append(row['y_start'] - timedelta(days=1))
        df['x_first'] = (df['y_end'].shift(1) + timedelta(days=1)).fillna(self.prices.index[0])
        df['x_last'] = xl

        self.x_predict_start = (df.iloc[-1]['y_end'] + timedelta(days=1)).date()

        return df

    def create_target_labels(self):
        target = []
        for index, row in self.agent_dates.iterrows():
            returns = sum(self.log_returns.loc[row['y_start']: row['y_end']]) * self.direction
            target.append([1 if returns > 0 else 0][0])

        self.agent_dates['target'] = target

        return

    def create_features_df(self):
        self.features = Feature_Generator.add_features(self.log_returns.to_frame(), self.asset)
        return

    def split_test_train(self):
        dates = self.agent_dates
        features = self.features

        X = []
        y = []
        for i, row in dates.iterrows():
            if len(features.loc[row['x_first']:row['x_last']]) < self.features_window:
                continue
            y.append(row['target'])
            trailing_features = features.loc[row['x_first']:row['x_last']][-self.features_window:]
            X.append(np.hstack(trailing_features.values))
            # X.append(trailing_features.values) for CNN/Transformer

        assert len(y) == len(X)
        self.X = np.array(X)
        self.y = np.array(y)
        train_size = round(self.train_test_percent * len(X))
        self.y_train, self.X_train = y[:train_size], X[:train_size]
        self.y_test, self.X_test = y[train_size:], X[train_size:]

        return

    def feature_selection(self):
        if self.feature_sel == 'L1':
            lsvc = LinearSVC(C=0.1,
                             penalty="l1",
                             dual=False)
            lsvc.fit(self.X,
                     self.y)
            self.feature_model = SelectFromModel(lsvc,
                                                 prefit=True)
            self.X_new = self.feature_model.transform(self.X)

        elif self.feature_sel == 'Tree':
            feature_extractor = ExtraTreesClassifier(n_estimators=50)
            feature_extractor.fit(self.X,
                                  self.y)
            self.feature_model = SelectFromModel(feature_extractor,
                                                 prefit=True)
            self.X_new = self.feature_model.transform(self.X)

        assert self.X_new.size != 0

        return

    def train_classifier(self):

        if self.HP_opt == 'SK_CV':

            param_grid = {'alpha': [0, 0.001, 0.005, 0.01, 0.05],
                          'lambda': [1, 1.5, 2],
                          'gamma': [i / 10.0 for i in range(0, 5)],
                          'max_depth': range(3, 6),
                          'min_child_weight': range(1, 5),
                          'subsample': [0.5, 0.75, 1],
                          'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1],
                          'scale_pos_weight': [0, 1]
                          }

            optimizer = RandomizedSearchCV(self.classifier,
                                           param_distributions=param_grid,
                                           n_jobs=-1,
                                           cv=self.folds)

            optimizer.fit(self.X_new,
                          self.y)

            base_model = optimizer.best_estimator_
            logger.info("{}".format(optimizer.best_params_))

            self.model = CalibratedClassifierCV(base_estimator=base_model,
                                                cv="prefit")
            self.model.fit(X=self.X_new,
                           y=self.y)

        elif self.HP_opt == 'Hyperopt':

            obj = HPOpt(self.X_new, self.y)
            best = obj.process()
            hyperopt_params = space_eval(obj.xgb_para, best)

            base_model = self.classifier
            base_model.kwargs = hyperopt_params['reg_params']

            self.model = CalibratedClassifierCV(base_estimator=base_model,
                                                cv=self.folds)
            self.model.fit(X=self.X_new,
                           y=self.y)

        else:
            base_model = self.classifier

            self.model = CalibratedClassifierCV(base_estimator=base_model,
                                                cv=self.folds)
            self.model.fit(X=self.X_new,
                           y=self.y)

        self.training_score = self.model.score(self.X_new, self.y)
        self.roc_auc = roc_auc_score(self.y, self.model.predict(self.X_new))

        return

    def train_CNN(self):
        self.model = self.make_cnn_model(input_shape=self.X.shape[1:])

        epochs = 500
        batch_size = 32

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],)

        self.model.fit(
            self.X,
            self.y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            #verbose=1,
        )

        return

    def train_transformer(self, epochs=250, batch_size=64):
        input_shape = self.X.shape[1:]

        self.model = self.build_transformer_model(
            input_shape,
            head_size=256,
            num_heads=4,
            ff_dim=4,
            num_transformer_blocks=4,
            mlp_units=[128],
            mlp_dropout=0.4,
            dropout=0.25,
        )

        self.model.compile(loss="sparse_categorical_crossentropy",
                           optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                           metrics=["sparse_categorical_accuracy"],)

        self.model.fit(self.X, self.y, validation_split=0.2, epochs=epochs, batch_size=batch_size)

        return

    def predict(self):
        predict_x_raw = self.features[self.x_predict_start:self.x_predict_end][-self.features_window:].values
        prediction_data = np.hstack(predict_x_raw).reshape(1, -1)

        # prediction = self.model.predict(predict_x_raw.reshape(-1, 10, 20))[0]  - For CNN/Transformer
        prediction_data = self.feature_model.transform(prediction_data)
        try:
            prediction = self.model.predict(prediction_data)[0]
        except ValueError:
            return None, None

        probabilities = self.model.predict_proba(prediction_data)[0]

        return prediction, probabilities

    def build_transformer_model(self, input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
        inputs = keras.Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = self.make_transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(2, activation="softmax")(x)
        return keras.Model(inputs, outputs)

    @staticmethod
    def make_cnn_model(input_shape):
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.ReLU()(conv1)

        conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.ReLU()(conv2)

        conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.ReLU()(conv3)

        gap = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(2, activation="softmax")(gap)

        return keras.models.Model(inputs=input_layer, outputs=output_layer)

    @staticmethod
    def make_transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
        x = layers.Dropout(dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res


def create_classifier(dates, prices, x_predict_end, direction):
    classifier = BinaryClassifier(dates, prices, x_predict_end, direction)
    classifier.create_target_labels()
    classifier.create_features_df()
    classifier.split_test_train()
    classifier.feature_selection()

    classifier.train_classifier()
    # classifier.train_CNN()
    # classifier.train_transformer()

    return classifier
