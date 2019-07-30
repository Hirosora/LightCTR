import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

from models.DeepFM import DeepFM
from models.WideAndDeep import WideAndDeep
from models.FNN import FNN
from models.PNN import PNN
from models.DCN import DCN
from models.xDeepFM import xDeepFM
from models.NFM import NFM
from models.AFM import AFM
from models.AutoInt import AutoInt
from core.features import FeatureMetas

if __name__ == "__main__":

    # Read dataset
    data = pd.read_csv('../datasets/avazu_1w.txt')

    # Get columns' names
    sparse_features = list(data.columns)
    target = ['click']

    # Preprocess your data
    data[sparse_features] = data[sparse_features].fillna('-1', )
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # Split your dataset
    train, test = train_test_split(data, test_size=0.2)
    train_model_input = {name: train[name] for name in sparse_features}
    test_model_input = {name: test[name] for name in sparse_features}

    # Instantiate a FeatureMetas object, add your features' meta information to it
    feature_metas = FeatureMetas()
    for feat in sparse_features:
        feature_metas.add_sparse_feature(name=feat, one_hot_dim=data[feat].nunique(), embedding_dim=32)

    # Instantiate a model and compile it
    model = DeepFM(
        feature_metas=feature_metas,
        linear_slots=sparse_features,
        fm_slots=sparse_features,
        dnn_slots=sparse_features
    )
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=['binary_crossentropy'])

    # Train the model
    history = model.fit(x=train_model_input,
                        y=train[target].values,
                        batch_size=128,
                        epochs=1,
                        verbose=2,
                        validation_split=0.2)

    # Testing
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
