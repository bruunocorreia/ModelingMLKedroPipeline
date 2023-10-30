"""
This is a boilerplate pipeline
generated using Kedro 0.18.14
"""

from kedro.pipeline import Pipeline, node, pipeline

from .pre_process.train_test_split_node import train_test_split_dist
from .feature_selection.feature_selection_node import UnsupervisedFeatureSelection,SupervisedFeatureSelection
from .model_training.model_train_node import train_and_save_xgboost_model
from .model_training.model_summary_node import load_xgboost_model_and_get_feature_importance
from .model_training.model_train_evaluation_node import EvaluateModel
from .model_scoring.model_scoring_node import scoring_dataframe

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_test_split_dist,
                inputs=["train","params:model_options"],
                outputs=["x_train", "x_test", "y_train", "y_test"],
                tags="split",
            ),
            node(
                func=UnsupervisedFeatureSelection,
                inputs=["x_train","params:model_feature_selection"],
                outputs="features_unsupervised",
                tags="feature_selection_unsupervised",
            ),
            node(
                func=SupervisedFeatureSelection,
                inputs=["x_train","y_train", "features_unsupervised"],
                outputs="features_supervised",
                tags="feature_selection_supervised",
            ),
             node(
                func=train_and_save_xgboost_model,
                inputs=["x_train", "y_train","features_supervised"],
                outputs="model_training",
                tags="model_training",
            ),
             node(
                func=load_xgboost_model_and_get_feature_importance,
                inputs=["model_training"],
                outputs="model_summary",
                tags="model_summary",
            ),
             node(
                func=EvaluateModel,
                inputs=["model_training", "x_train", "y_train", "x_test", "y_test", "features_supervised"],
                outputs="model_evaluation",
                tags="model_evaluation",
            ),
            node(
                func=scoring_dataframe,
                inputs=["validation", "features_supervised", "model_training","params:model_options"],
                outputs="scoring",
                tags="scoring",
            ),
        ]
    )









# from kedro.pipeline import Pipeline, node, pipeline

# from ..pre_process import train_test_split_node as split
# from ..feature_selection import feature_selection_node as fs
# from ..model_training import model_summary_node as ms, model_train_evaluation_node as me, model_train_node as mt
# from ..model_scoring import model_scoring_node as sc

# def create_pipeline(**kwargs) -> Pipeline:
#     return pipeline(
#         [
#             node(
#                 func=train_test_split_dist,
#                 inputs=["train","params:model_options"],
#                 outputs=["x_train", "x_test", "y_train", "y_test"],
#                 name="split",
#             ),
#         ]
#     )



# def create_pipeline(**kwargs) -> Pipeline:
#     return pipeline(
#         [split.
#             node(
#                 func=train_test_split_dist,
#                 inputs=["train", "params: model_options"],
#                 outputs=["x_train", "x_test", "y_train", "y_test"],
#                 name="split",
#             ),
            # node(
            #     func=UnsupervisedFeatureSelection,
            #     inputs=["x_train", "variance", "corr"],
            #     outputs="features_unsupervised",
            #     name="feature_selection_unsupervised",
            # ),
            # node(
            #     func=SupervisedFeatureSelection,
            #     inputs=["x_train","y_train", "features"],
            #     outputs="features_supervised",
            #     name="feature_selection_supervised",
            # ),
            #  node(
            #     func=train_and_save_xgboost_model,
            #     inputs=["x_train", "y_train","features"],
            #     outputs="model_training",
            #     name="model_training",
            # ),
            #  node(
            #     func=load_xgboost_model_and_get_feature_importance,
            #     inputs=["model"],
            #     outputs="model_summary",
            #     name="model_summary",
            # ),
            #  node(
            #     func=EvaluateModel,
            #     inputs=["model", "x_train", "y_train", "x_test", "y_test", "features"],
            #     outputs="model_evaluation",
            #     name="model_evaluation",
            # ),
            # node(
            #     func=scoring_dataframe,
            #     inputs=["validation", "id_columns", "features", "model"],
            #     outputs="scoring",
            #     name="scoring",
            # ),
    #     ]
    # )
