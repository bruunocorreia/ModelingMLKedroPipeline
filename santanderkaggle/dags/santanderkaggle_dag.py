from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

from kedro.framework.session import KedroSession
from kedro.framework.project import configure_project


class KedroOperator(BaseOperator):
    @apply_defaults
    def __init__(
        self,
        package_name: str,
        pipeline_name: str,
        node_name: str,
        project_path: str | Path,
        env: str,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.package_name = package_name
        self.pipeline_name = pipeline_name
        self.node_name = node_name
        self.project_path = project_path
        self.env = env

    def execute(self, context):
        configure_project(self.package_name)
        with KedroSession.create(self.package_name,
                                 self.project_path,
                                 env=self.env) as session:
            session.run(self.pipeline_name, node_names=[self.node_name])


# Kedro settings required to run your pipeline
env = "airflow"
pipeline_name = "__default__"
project_path = Path.cwd()
package_name = "santanderkaggle"

# Using a DAG context manager, you don't have to specify the dag property of each task
with DAG(
    dag_id="santanderkaggle",
    start_date=datetime(2023,1,1),
    max_active_runs=3,
    # https://airflow.apache.org/docs/stable/scheduler.html#dag-runs
    schedule_interval="@once",
    catchup=False,
    # Default settings applied to all tasks
    default_args=dict(
        owner="airflow",
        depends_on_past=False,
        email_on_failure=False,
        email_on_retry=False,
        retries=1,
        retry_delay=timedelta(minutes=5)
    )
) as dag:
    tasks = {
        "train-test-split-dist-train-params-model-options-x-train-x-test-y-train-y-test": KedroOperator(
            task_id="train-test-split-dist-train-params-model-options-x-train-x-test-y-train-y-test",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="train_test_split_dist([train,params:model_options]) -> [x_train,x_test,y_train,y_test]",
            project_path=project_path,
            env=env,
        ),
        "unsupervisedfeatureselection-x-train-params-model-feature-selection-features-unsupervised": KedroOperator(
            task_id="unsupervisedfeatureselection-x-train-params-model-feature-selection-features-unsupervised",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="UnsupervisedFeatureSelection([x_train,params:model_feature_selection]) -> [features_unsupervised]",
            project_path=project_path,
            env=env,
        ),
        "supervisedfeatureselection-x-train-y-train-features-unsupervised-features-supervised": KedroOperator(
            task_id="supervisedfeatureselection-x-train-y-train-features-unsupervised-features-supervised",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="SupervisedFeatureSelection([x_train,y_train,features_unsupervised]) -> [features_supervised]",
            project_path=project_path,
            env=env,
        ),
        "train-and-save-xgboost-model-x-train-y-train-features-supervised-model-training": KedroOperator(
            task_id="train-and-save-xgboost-model-x-train-y-train-features-supervised-model-training",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="train_and_save_xgboost_model([x_train,y_train,features_supervised]) -> [model_training]",
            project_path=project_path,
            env=env,
        ),
        "evaluatemodel-model-training-x-train-y-train-x-test-y-test-features-supervised-model-evaluation": KedroOperator(
            task_id="evaluatemodel-model-training-x-train-y-train-x-test-y-test-features-supervised-model-evaluation",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="EvaluateModel([model_training,x_train,y_train,x_test,y_test,features_supervised]) -> [model_evaluation]",
            project_path=project_path,
            env=env,
        ),
        "load-xgboost-model-and-get-feature-importance-model-training-model-summary": KedroOperator(
            task_id="load-xgboost-model-and-get-feature-importance-model-training-model-summary",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="load_xgboost_model_and_get_feature_importance([model_training]) -> [model_summary]",
            project_path=project_path,
            env=env,
        ),
        "scoring-dataframe-validation-features-supervised-model-training-params-model-options-scoring": KedroOperator(
            task_id="scoring-dataframe-validation-features-supervised-model-training-params-model-options-scoring",
            package_name=package_name,
            pipeline_name=pipeline_name,
            node_name="scoring_dataframe([validation,features_supervised,model_training,params:model_options]) -> [scoring]",
            project_path=project_path,
            env=env,
        ),
    }

    tasks["train-test-split-dist-train-params-model-options-x-train-x-test-y-train-y-test"] >> tasks["unsupervisedfeatureselection-x-train-params-model-feature-selection-features-unsupervised"]
    tasks["train-test-split-dist-train-params-model-options-x-train-x-test-y-train-y-test"] >> tasks["supervisedfeatureselection-x-train-y-train-features-unsupervised-features-supervised"]
    tasks["train-test-split-dist-train-params-model-options-x-train-x-test-y-train-y-test"] >> tasks["train-and-save-xgboost-model-x-train-y-train-features-supervised-model-training"]
    tasks["train-test-split-dist-train-params-model-options-x-train-x-test-y-train-y-test"] >> tasks["evaluatemodel-model-training-x-train-y-train-x-test-y-test-features-supervised-model-evaluation"]
    tasks["unsupervisedfeatureselection-x-train-params-model-feature-selection-features-unsupervised"] >> tasks["supervisedfeatureselection-x-train-y-train-features-unsupervised-features-supervised"]
    tasks["supervisedfeatureselection-x-train-y-train-features-unsupervised-features-supervised"] >> tasks["train-and-save-xgboost-model-x-train-y-train-features-supervised-model-training"]
    tasks["supervisedfeatureselection-x-train-y-train-features-unsupervised-features-supervised"] >> tasks["evaluatemodel-model-training-x-train-y-train-x-test-y-test-features-supervised-model-evaluation"]
    tasks["supervisedfeatureselection-x-train-y-train-features-unsupervised-features-supervised"] >> tasks["scoring-dataframe-validation-features-supervised-model-training-params-model-options-scoring"]
    tasks["train-and-save-xgboost-model-x-train-y-train-features-supervised-model-training"] >> tasks["load-xgboost-model-and-get-feature-importance-model-training-model-summary"]
    tasks["train-and-save-xgboost-model-x-train-y-train-features-supervised-model-training"] >> tasks["evaluatemodel-model-training-x-train-y-train-x-test-y-test-features-supervised-model-evaluation"]
    tasks["train-and-save-xgboost-model-x-train-y-train-features-supervised-model-training"] >> tasks["scoring-dataframe-validation-features-supervised-model-training-params-model-options-scoring"]
