"""
MLflow integration for experiment tracking.

This module provides MLflow integration for tracking experiments,
models, and artifacts in Quant Lab.
"""

import os
import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime
import json

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Install with: pip install mlflow")

from ..strategies import StrategySpec
from ..core.config import get_config

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    MLflow integration for experiment tracking.
    
    This class provides a unified interface for tracking experiments,
    logging parameters, metrics, and artifacts in MLflow.
    """
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "quant-lab",
        artifact_location: Optional[str] = None
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking URI (defaults to local file system)
            experiment_name: Name of the MLflow experiment
            artifact_location: Location for storing artifacts
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is required but not installed. Install with: pip install mlflow")
        
        self.tracking_uri = tracking_uri or "file:./mlruns"
        self.experiment_name = experiment_name
        self.artifact_location = artifact_location
        
        # Configure MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Get or create experiment
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if self.experiment is None:
            self.experiment = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=artifact_location
            )
        
        mlflow.set_experiment(experiment_name)
        
        # Current run tracking
        self.current_run = None
        self.run_id = None
        
        logger.info(f"MLflow tracker initialized for experiment: {experiment_name}")
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Additional tags for the run
            description: Run description
            
        Returns:
            Run ID
        """
        if self.current_run is not None:
            logger.warning("Run already active. Ending current run first.")
            self.end_run()
        
        # Set default tags
        default_tags = {
            "project": "quant-lab",
            "version": "0.1.0",
            "created_at": datetime.now().isoformat()
        }
        
        if tags:
            default_tags.update(tags)
        
        # Start run
        self.current_run = mlflow.start_run(
            run_name=run_name,
            tags=default_tags,
            description=description
        )
        
        self.run_id = self.current_run.info.run_id
        logger.info(f"Started MLflow run: {run_name or self.run_id}")
        
        return self.run_id
    
    def end_run(self, status: str = "FINISHED"):
        """
        End the current MLflow run.
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if self.current_run is not None:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run: {self.run_id} with status: {status}")
            self.current_run = None
            self.run_id = None
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to the current run.
        
        Args:
            params: Dictionary of parameters to log
        """
        if self.current_run is None:
            logger.warning("No active run. Cannot log parameters.")
            return
        
        # Convert parameters to strings for MLflow compatibility
        string_params = {}
        for key, value in params.items():
            if isinstance(value, (dict, list)):
                string_params[key] = json.dumps(value)
            else:
                string_params[key] = str(value)
        
        mlflow.log_params(string_params)
        logger.debug(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to the current run.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number for the metrics
        """
        if self.current_run is None:
            logger.warning("No active run. Cannot log metrics.")
            return
        
        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged {len(metrics)} metrics at step {step}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact to the current run.
        
        Args:
            local_path: Path to the local file
            artifact_path: Path within the artifact directory
        """
        if self.current_run is None:
            logger.warning("No active run. Cannot log artifacts.")
            return
        
        if os.path.exists(local_path):
            mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Logged artifact: {local_path}")
        else:
            logger.warning(f"Artifact file not found: {local_path}")
    
    def log_model(
        self,
        model,
        artifact_path: str,
        model_type: str = "sklearn",
        **kwargs
    ):
        """
        Log a trained model to the current run.
        
        Args:
            model: Trained model object
            artifact_path: Path within the artifact directory
            model_type: Type of model (sklearn, pytorch, etc.)
            **kwargs: Additional arguments for model logging
        """
        if self.current_run is None:
            logger.warning("No active run. Cannot log model.")
            return
        
        try:
            if model_type == "sklearn":
                mlflow.sklearn.log_model(model, artifact_path, **kwargs)
            elif model_type == "pytorch":
                mlflow.pytorch.log_model(model, artifact_path, **kwargs)
            else:
                # Generic model logging
                mlflow.log_artifact(model, artifact_path)
            
            logger.info(f"Logged {model_type} model to {artifact_path}")
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
    
    def log_strategy_spec(self, strategy_spec: StrategySpec):
        """
        Log strategy specification to the current run.
        
        Args:
            strategy_spec: Strategy specification object
        """
        if self.current_run is None:
            logger.warning("No active run. Cannot log strategy spec.")
            return
        
        # Log strategy parameters
        strategy_params = {
            "strategy_name": strategy_spec.name,
            "universe_size": len(strategy_spec.universe),
            "start_date": strategy_spec.start_date.isoformat() if strategy_spec.start_date else None,
            "end_date": strategy_spec.end_date.isoformat() if strategy_spec.end_date else None,
            "target_column": strategy_spec.target_column,
            "holding_period": strategy_spec.holding_period,
            "max_positions": strategy_spec.max_positions
        }
        
        self.log_params(strategy_params)
        
        # Log full strategy spec as artifact
        spec_file = f"strategy_spec_{strategy_spec.name}.json"
        with open(spec_file, 'w') as f:
            json.dump(strategy_spec.model_dump(), f, indent=2, default=str)
        
        self.log_artifact(spec_file)
        os.remove(spec_file)  # Clean up temporary file
        
        logger.info(f"Logged strategy spec for: {strategy_spec.name}")
    
    def log_backtest_results(self, results: Dict[str, Any], results_file: str):
        """
        Log backtest results to the current run.
        
        Args:
            results: Backtest results dictionary
            results_file: Path to the results file
        """
        if self.current_run is None:
            logger.warning("No active run. Cannot log backtest results.")
            return
        
        # Log key metrics
        if 'metrics' in results:
            self.log_metrics(results['metrics'])
        
        # Log results file as artifact
        self.log_artifact(results_file, "backtest_results")
        
        logger.info("Logged backtest results")
    
    def log_feature_importance(self, feature_importance: Dict[str, float]):
        """
        Log feature importance scores to the current run.
        
        Args:
            feature_importance: Dictionary of feature importance scores
        """
        if self.current_run is None:
            logger.warning("No active run. Cannot log feature importance.")
            return
        
        # Convert to metrics format
        importance_metrics = {}
        for feature, importance in feature_importance.items():
            metric_name = f"feature_importance_{feature}"
            importance_metrics[metric_name] = float(importance)
        
        self.log_metrics(importance_metrics)
        logger.info(f"Logged feature importance for {len(feature_importance)} features")
    
    def get_run_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the current run.
        
        Returns:
            Dictionary with run information or None if no active run
        """
        if self.current_run is None:
            return None
        
        return {
            "run_id": self.run_id,
            "experiment_id": self.current_run.info.experiment_id,
            "status": self.current_run.info.status,
            "start_time": self.current_run.info.start_time,
            "end_time": self.current_run.info.end_time,
            "artifact_uri": self.current_run.info.artifact_uri
        }
    
    def search_runs(
        self,
        filter_string: Optional[str] = None,
        max_results: int = 100
    ) -> list:
        """
        Search for previous runs.
        
        Args:
            filter_string: MLflow filter string
            max_results: Maximum number of results to return
            
        Returns:
            List of run information
        """
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                filter_string=filter_string,
                max_results=max_results
            )
            return runs.to_dict('records')
        except Exception as e:
            logger.error(f"Failed to search runs: {e}")
            return []
    
    def load_model(self, run_id: str, artifact_path: str, model_type: str = "sklearn"):
        """
        Load a logged model from a specific run.
        
        Args:
            run_id: MLflow run ID
            artifact_path: Path to the model artifact
            model_type: Type of model to load
            
        Returns:
            Loaded model object
        """
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            
            if model_type == "sklearn":
                return mlflow.sklearn.load_model(model_uri)
            elif model_type == "pytorch":
                return mlflow.pytorch.load_model(model_uri)
            else:
                logger.warning(f"Unknown model type: {model_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    def compare_runs(self, run_ids: list, metric_names: list) -> Dict[str, Any]:
        """
        Compare multiple runs based on specified metrics.
        
        Args:
            run_ids: List of run IDs to compare
            metric_names: List of metric names to compare
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {}
        
        for run_id in run_ids:
            try:
                run = mlflow.get_run(run_id)
                run_metrics = run.data.metrics
                
                comparison[run_id] = {
                    "run_name": run.data.tags.get("mlflow.runName", run_id),
                    "metrics": {metric: run_metrics.get(metric, None) for metric in metric_names}
                }
            except Exception as e:
                logger.error(f"Failed to get run {run_id}: {e}")
                comparison[run_id] = {"error": str(e)}
        
        return comparison
    
    def export_results(self, output_dir: str = "mlflow_exports"):
        """
        Export all experiment results to a local directory.
        
        Args:
            output_dir: Directory to export results to
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Get all runs for the experiment
            runs = self.search_runs(max_results=1000)
            
            for run in runs:
                run_id = run.get('run_id')
                if run_id:
                    run_dir = os.path.join(output_dir, run_id)
                    os.makedirs(run_dir, exist_ok=True)
                    
                    # Export run data
                    run_data = {
                        "run_id": run_id,
                        "metrics": run.get('metrics', {}),
                        "params": run.get('params', {}),
                        "tags": run.get('tags', {})
                    }
                    
                    with open(os.path.join(run_dir, "run_info.json"), 'w') as f:
                        json.dump(run_data, f, indent=2)
            
            logger.info(f"Exported {len(runs)} runs to {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")


def create_mlflow_tracker(
    experiment_name: Optional[str] = None,
    tracking_uri: Optional[str] = None
) -> MLflowTracker:
    """
    Factory function to create MLflow tracker.
    
    Args:
        experiment_name: Name of the experiment
        tracking_uri: MLflow tracking URI
        
    Returns:
        Configured MLflow tracker instance
    """
    config = get_config()
    
    # Get MLflow configuration from config
    mlflow_config = config.get('mlflow', {})
    
    exp_name = experiment_name or mlflow_config.get('experiment_name', 'quant-lab')
    tracking_uri = tracking_uri or mlflow_config.get('tracking_uri', None)
    artifact_location = mlflow_config.get('artifact_location', None)
    
    return MLflowTracker(
        tracking_uri=tracking_uri,
        experiment_name=exp_name,
        artifact_location=artifact_location
    )
