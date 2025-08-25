"""
Experiment management for Quant Lab.

This module provides high-level experiment management capabilities
for strategy development and backtesting workflows.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import json
import pandas as pd

from .mlflow_tracker import MLflowTracker
from ..strategies import StrategySpec
from ..core.config import get_config

logger = logging.getLogger(__name__)


class ExperimentManager:
    """
    High-level experiment management for Quant Lab.
    
    This class provides a unified interface for managing experiments,
    tracking runs, and organizing results across different strategies.
    """
    
    def __init__(
        self,
        experiment_name: str = "quant-lab",
        base_output_dir: str = "runs",
        auto_track: bool = True
    ):
        """
        Initialize experiment manager.
        
        Args:
            experiment_name: Name of the MLflow experiment
            base_output_dir: Base directory for experiment outputs
            auto_track: Whether to automatically track experiments
        """
        self.experiment_name = experiment_name
        self.base_output_dir = base_output_dir
        self.auto_track = auto_track
        
        # Create output directories
        os.makedirs(base_output_dir, exist_ok=True)
        os.makedirs(os.path.join(base_output_dir, "strategies"), exist_ok=True)
        os.makedirs(os.path.join(base_output_dir, "backtests"), exist_ok=True)
        os.makedirs(os.path.join(base_output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(base_output_dir, "reports"), exist_ok=True)
        
        # Initialize MLflow tracker if auto_track is enabled
        self.tracker = None
        if auto_track:
            try:
                self.tracker = MLflowTracker(experiment_name=experiment_name)
                logger.info(f"MLflow tracking enabled for experiment: {experiment_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize MLflow tracking: {e}")
                self.auto_track = False
        
        # Experiment registry
        self.experiments = {}
        self.current_experiment = None
        
        logger.info(f"Experiment manager initialized: {experiment_name}")
    
    def start_strategy_experiment(
        self,
        strategy_spec: StrategySpec,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Start a new strategy experiment.
        
        Args:
            strategy_spec: Strategy specification
            description: Experiment description
            tags: Additional tags
            
        Returns:
            Experiment ID
        """
        experiment_id = f"strategy_{strategy_spec.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create experiment directory
        exp_dir = os.path.join(self.base_output_dir, "strategies", experiment_id)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Initialize experiment
        experiment = {
            "id": experiment_id,
            "type": "strategy",
            "strategy_spec": strategy_spec,
            "start_time": datetime.now(),
            "status": "running",
            "output_dir": exp_dir,
            "description": description,
            "tags": tags or {},
            "runs": [],
            "results": {}
        }
        
        self.experiments[experiment_id] = experiment
        self.current_experiment = experiment_id
        
        # Start MLflow run if tracking is enabled
        if self.tracker and self.auto_track:
            try:
                run_id = self.tracker.start_run(
                    run_name=f"strategy_{strategy_spec.name}",
                    description=description,
                    tags=tags
                )
                
                # Log strategy specification
                self.tracker.log_strategy_spec(strategy_spec)
                
                # Store run ID
                experiment["mlflow_run_id"] = run_id
                
                logger.info(f"Started strategy experiment: {experiment_id} (MLflow run: {run_id})")
            except Exception as e:
                logger.error(f"Failed to start MLflow run: {e}")
        
        # Save experiment metadata
        self._save_experiment_metadata(experiment_id)
        
        logger.info(f"Started strategy experiment: {experiment_id}")
        return experiment_id
    
    def start_backtest_experiment(
        self,
        strategy_name: str,
        backtest_config: Dict[str, Any],
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Start a new backtest experiment.
        
        Args:
            strategy_name: Name of the strategy being backtested
            backtest_config: Backtest configuration
            description: Experiment description
            tags: Additional tags
            
        Returns:
            Experiment ID
        """
        experiment_id = f"backtest_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create experiment directory
        exp_dir = os.path.join(self.base_output_dir, "backtests", experiment_id)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Initialize experiment
        experiment = {
            "id": experiment_id,
            "type": "backtest",
            "strategy_name": strategy_name,
            "backtest_config": backtest_config,
            "start_time": datetime.now(),
            "status": "running",
            "output_dir": exp_dir,
            "description": description,
            "tags": tags or {},
            "runs": [],
            "results": {}
        }
        
        self.experiments[experiment_id] = experiment
        self.current_experiment = experiment_id
        
        # Start MLflow run if tracking is enabled
        if self.tracker and self.auto_track:
            try:
                run_id = self.tracker.start_run(
                    run_name=f"backtest_{strategy_name}",
                    description=description,
                    tags=tags
                )
                
                # Log backtest configuration
                self.tracker.log_params(backtest_config)
                
                # Store run ID
                experiment["mlflow_run_id"] = run_id
                
                logger.info(f"Started backtest experiment: {experiment_id} (MLflow run: {run_id})")
            except Exception as e:
                logger.error(f"Failed to start MLflow run: {e}")
        
        # Save experiment metadata
        self._save_experiment_metadata(experiment_id)
        
        logger.info(f"Started backtest experiment: {experiment_id}")
        return experiment_id
    
    def log_experiment_step(
        self,
        step_name: str,
        step_data: Dict[str, Any],
        step_type: str = "info"
    ):
        """
        Log a step in the current experiment.
        
        Args:
            step_name: Name of the step
            step_data: Step data to log
            step_type: Type of step (info, warning, error)
        """
        if not self.current_experiment:
            logger.warning("No active experiment. Cannot log step.")
            return
        
        experiment = self.experiments[self.current_experiment]
        
        step = {
            "name": step_name,
            "type": step_type,
            "timestamp": datetime.now().isoformat(),
            "data": step_data
        }
        
        experiment["runs"].append(step)
        
        # Log to MLflow if tracking is enabled
        if self.tracker and self.auto_track:
            try:
                if step_type == "metrics":
                    self.tracker.log_metrics(step_data)
                elif step_type == "params":
                    self.tracker.log_params(step_data)
                else:
                    # Log as artifact
                    step_file = os.path.join(experiment["output_dir"], f"{step_name}.json")
                    with open(step_file, 'w') as f:
                        json.dump(step_data, f, indent=2, default=str)
                    self.tracker.log_artifact(step_file, f"steps/{step_name}")
            except Exception as e:
                logger.error(f"Failed to log step to MLflow: {e}")
        
        # Save updated experiment metadata
        self._save_experiment_metadata(self.current_experiment)
        
        logger.debug(f"Logged experiment step: {step_name}")
    
    def log_experiment_results(
        self,
        results: Dict[str, Any],
        results_type: str = "final"
    ):
        """
        Log final results for the current experiment.
        
        Args:
            results: Experiment results
            results_type: Type of results (final, intermediate)
        """
        if not self.current_experiment:
            logger.warning("No active experiment. Cannot log results.")
            return
        
        experiment = self.experiments[self.current_experiment]
        
        # Store results
        experiment["results"][results_type] = {
            "timestamp": datetime.now().isoformat(),
            "data": results
        }
        
        # Save results to file
        results_file = os.path.join(
            experiment["output_dir"],
            f"{results_type}_results.json"
        )
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Log to MLflow if tracking is enabled
        if self.tracker and self.auto_track:
            try:
                if results_type == "final":
                    # Log final metrics
                    if "metrics" in results:
                        self.tracker.log_metrics(results["metrics"])
                    
                    # Log results file
                    self.tracker.log_artifact(results_file, f"{results_type}_results")
                    
                    # End MLflow run
                    self.tracker.end_run()
                    
                    logger.info(f"Completed experiment: {self.current_experiment}")
            except Exception as e:
                logger.error(f"Failed to log results to MLflow: {e}")
        
        # Update experiment status
        if results_type == "final":
            experiment["status"] = "completed"
            experiment["end_time"] = datetime.now()
        
        # Save updated experiment metadata
        self._save_experiment_metadata(self.current_experiment)
        
        logger.info(f"Logged {results_type} results for experiment: {self.current_experiment}")
    
    def end_experiment(self, status: str = "completed"):
        """
        End the current experiment.
        
        Args:
            status: Final experiment status
        """
        if not self.current_experiment:
            logger.warning("No active experiment to end.")
            return
        
        experiment = self.experiments[self.current_experiment]
        experiment["status"] = status
        experiment["end_time"] = datetime.now()
        
        # End MLflow run if tracking is enabled
        if self.tracker and self.auto_track and experiment.get("mlflow_run_id"):
            try:
                self.tracker.end_run(status=status.upper())
            except Exception as e:
                logger.error(f"Failed to end MLflow run: {e}")
        
        # Save final experiment metadata
        self._save_experiment_metadata(self.current_experiment)
        
        logger.info(f"Ended experiment: {self.current_experiment} with status: {status}")
        
        # Clear current experiment
        self.current_experiment = None
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment information.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment information or None if not found
        """
        return self.experiments.get(experiment_id)
    
    def list_experiments(
        self,
        experiment_type: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List experiments with optional filtering.
        
        Args:
            experiment_type: Filter by experiment type
            status: Filter by experiment status
            
        Returns:
            List of experiment information
        """
        experiments = list(self.experiments.values())
        
        if experiment_type:
            experiments = [exp for exp in experiments if exp["type"] == experiment_type]
        
        if status:
            experiments = [exp for exp in experiments if exp["status"] == status]
        
        return experiments
    
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metric_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metric_names: List of metric names to compare
            
        Returns:
            Comparison results
        """
        comparison = {}
        
        for exp_id in experiment_ids:
            experiment = self.experiments.get(exp_id)
            if experiment:
                comparison[exp_id] = {
                    "name": experiment.get("strategy_name", exp_id),
                    "type": experiment["type"],
                    "status": experiment["status"],
                    "start_time": experiment["start_time"].isoformat(),
                    "end_time": experiment.get("end_time", "").isoformat(),
                    "metrics": experiment.get("results", {}).get("final", {}).get("data", {}).get("metrics", {})
                }
        
        # Create comparison summary
        if metric_names:
            summary = {}
            for metric in metric_names:
                values = []
                for exp_data in comparison.values():
                    if metric in exp_data["metrics"]:
                        values.append(exp_data["metrics"][metric])
                
                if values:
                    summary[metric] = {
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "values": values
                    }
            
            comparison["summary"] = summary
        
        return comparison
    
    def export_experiment(
        self,
        experiment_id: str,
        output_dir: Optional[str] = None
    ) -> str:
        """
        Export experiment to a directory.
        
        Args:
            experiment_id: Experiment ID to export
            output_dir: Output directory (defaults to experiment ID)
            
        Returns:
            Path to exported experiment
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        if output_dir is None:
            output_dir = f"export_{experiment_id}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy experiment files
        import shutil
        shutil.copytree(
            experiment["output_dir"],
            os.path.join(output_dir, "experiment_files"),
            dirs_exist_ok=True
        )
        
        # Export experiment metadata
        metadata_file = os.path.join(output_dir, "experiment_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(experiment, f, indent=2, default=str)
        
        # Export summary report
        summary_file = os.path.join(output_dir, "experiment_summary.md")
        self._generate_experiment_summary(experiment, summary_file)
        
        logger.info(f"Exported experiment {experiment_id} to {output_dir}")
        return output_dir
    
    def _save_experiment_metadata(self, experiment_id: str):
        """Save experiment metadata to file."""
        experiment = self.experiments[experiment_id]
        metadata_file = os.path.join(experiment["output_dir"], "experiment_metadata.json")
        
        # Convert datetime objects to strings for JSON serialization
        metadata = experiment.copy()
        for key in ["start_time", "end_time"]:
            if key in metadata and metadata[key]:
                metadata[key] = metadata[key].isoformat()
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _generate_experiment_summary(self, experiment: Dict[str, Any], output_file: str):
        """Generate a markdown summary of the experiment."""
        with open(output_file, 'w') as f:
            f.write(f"# Experiment Summary: {experiment['id']}\n\n")
            
            f.write(f"**Type:** {experiment['type']}\n")
            f.write(f"**Status:** {experiment['status']}\n")
            f.write(f"**Start Time:** {experiment['start_time']}\n")
            
            if experiment.get('end_time'):
                f.write(f"**End Time:** {experiment['end_time']}\n")
            
            if experiment.get('description'):
                f.write(f"**Description:** {experiment['description']}\n")
            
            f.write(f"\n## Results\n\n")
            
            for result_type, result_data in experiment.get('results', {}).items():
                f.write(f"### {result_type.title()} Results\n\n")
                f.write(f"**Timestamp:** {result_data['timestamp']}\n\n")
                
                if 'metrics' in result_data['data']:
                    f.write("**Metrics:**\n\n")
                    for metric, value in result_data['data']['metrics'].items():
                        f.write(f"- {metric}: {value}\n")
                    f.write("\n")
            
            f.write(f"## Steps\n\n")
            for step in experiment.get('runs', []):
                f.write(f"- **{step['name']}** ({step['type']}) - {step['timestamp']}\n")
    
    def cleanup_experiments(self, max_age_days: int = 30):
        """
        Clean up old experiments.
        
        Args:
            max_age_days: Maximum age of experiments to keep
        """
        cutoff_date = datetime.now() - pd.Timedelta(days=max_age_days)
        
        experiments_to_remove = []
        for exp_id, experiment in self.experiments.items():
            if experiment["start_time"] < cutoff_date:
                experiments_to_remove.append(exp_id)
        
        for exp_id in experiments_to_remove:
            try:
                # Remove experiment directory
                import shutil
                shutil.rmtree(experiment["output_dir"], ignore_errors=True)
                
                # Remove from registry
                del self.experiments[exp_id]
                
                logger.info(f"Cleaned up old experiment: {exp_id}")
            except Exception as e:
                logger.error(f"Failed to cleanup experiment {exp_id}: {e}")


def create_experiment_manager(
    experiment_name: Optional[str] = None,
    base_output_dir: Optional[str] = None,
    auto_track: bool = True
) -> ExperimentManager:
    """
    Factory function to create experiment manager.
    
    Args:
        experiment_name: Name of the experiment
        base_output_dir: Base output directory
        auto_track: Whether to enable automatic tracking
        
    Returns:
        Configured experiment manager instance
    """
    config = get_config()
    
    # Get configuration
    exp_name = experiment_name or config.get('experiments', {}).get('name', 'quant-lab')
    output_dir = base_output_dir or config.get('experiments', {}).get('output_dir', 'runs')
    tracking = auto_track and config.get('experiments', {}).get('auto_track', True)
    
    return ExperimentManager(
        experiment_name=exp_name,
        base_output_dir=output_dir,
        auto_track=tracking
    )
