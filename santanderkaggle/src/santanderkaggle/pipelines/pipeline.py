"""
This is a boilerplate pipeline
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from .pre_process import train_test_split_node
from .feature_selection import feature_selection_node
from .model_training import model_summary_node, model_train_evaluation_node, model_train_node
from .model_scoring import model_scoring_node

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_test_split_dist,
                inputs=["train","id_columns", "target_column", "test_size"],
                outputs=["x_train", "x_test", "y_train", "y_test"],
                name="split",
            ),
            node(
                func=UnsupervisedFeatureSelection,
                inputs=["x_train", "variance", "corr"],
                outputs="features_unsupervised",
                name="feature_selection_unsupervised",
            ),
            node(
                func=SupervisedFeatureSelection,
                inputs=["x_train","y_train", "features"],
                outputs="features_supervised",
                name="feature_selection_supervised",
            ),
             node(
                func=train_and_save_xgboost_model,
                inputs=["x_train", "y_train","features"],
                outputs="model_training",
                name="model_training",
            ),
             node(
                func=load_xgboost_model_and_get_feature_importance,
                inputs=["model"],
                outputs="model_summary",
                name="model_summary",
            ),
             node(
                func=EvaluateModel,
                inputs=["model", "x_train", "y_train", "x_test", "y_test", "features"],
                outputs="model_evaluation",
                name="model_evaluation",
            ),
            node(
                func=scoring_dataframe,
                inputs=["validation", "id_columns", "features", "model"],
                outputs="scoring",
                name="scoring",
            ),
        ]
    )
