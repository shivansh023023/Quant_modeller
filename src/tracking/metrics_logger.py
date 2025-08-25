"""
Metrics logging for Quant Lab experiments.

This module provides comprehensive metrics logging capabilities
for tracking performance, risk, and other metrics during experiments.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import json
import pandas as pd
import numpy as np

from .mlflow_tracker import MLflowTracker
from ..core.metrics import PerformanceMetrics, RiskMetrics

logger = logging.getLogger(__name__)


class MetricsLogger:
    """
    Comprehensive metrics logging for Quant Lab experiments.
    
    This class provides methods for logging various types of metrics
    including performance, risk, feature importance, and custom metrics.
    """
    
    def __init__(
        self,
        output_dir: str = "metrics_logs",
        auto_save: bool = True,
        mlflow_tracker: Optional[MLflowTracker] = None
    ):
        """
        Initialize metrics logger.
        
        Args:
            output_dir: Directory to save metrics logs
            auto_save: Whether to automatically save metrics to files
            mlflow_tracker: Optional MLflow tracker for experiment tracking
        """
        self.output_dir = output_dir
        self.auto_save = auto_save
        self.mlflow_tracker = mlflow_tracker
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize metrics calculators
        self.performance_calculator = PerformanceMetrics()
        self.risk_calculator = RiskMetrics()
        
        # Metrics storage
        self.metrics_history = {}
        self.current_session = None
        
        logger.info(f"Metrics logger initialized with output directory: {output_dir}")
    
    def start_session(self, session_name: str, session_metadata: Optional[Dict[str, Any]] = None):
        """
        Start a new metrics logging session.
        
        Args:
            session_name: Name of the session
            session_metadata: Additional metadata for the session
        """
        session_id = f"{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = {
            "id": session_id,
            "name": session_name,
            "start_time": datetime.now(),
            "metadata": session_metadata or {},
            "metrics": {},
            "timestamps": []
        }
        
        # Create session directory
        session_dir = os.path.join(self.output_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        self.current_session["output_dir"] = session_dir
        
        logger.info(f"Started metrics session: {session_id}")
    
    def end_session(self, save_results: bool = True):
        """
        End the current metrics session.
        
        Args:
            save_results: Whether to save session results
        """
        if not self.current_session:
            logger.warning("No active session to end.")
            return
        
        self.current_session["end_time"] = datetime.now()
        self.current_session["duration"] = (
            self.current_session["end_time"] - self.current_session["start_time"]
        ).total_seconds()
        
        if save_results:
            self._save_session_results()
        
        # Store session in history
        self.metrics_history[self.current_session["id"]] = self.current_session
        
        logger.info(f"Ended metrics session: {self.current_session['id']}")
        
        # Clear current session
        self.current_session = None
    
    def log_performance_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        step: Optional[int] = None,
        save_to_mlflow: bool = True
    ):
        """
        Log performance metrics for a series of returns.
        
        Args:
            returns: Series of returns
            benchmark_returns: Optional benchmark returns for comparison
            step: Step number for the metrics
            save_to_mlflow: Whether to save metrics to MLflow
        """
        if not self.current_session:
            logger.warning("No active session. Cannot log performance metrics.")
            return
        
        # Calculate performance metrics
        performance_metrics = self.performance_calculator.calculate_all_metrics(
            returns, benchmark_returns
        )
        
        # Store metrics
        timestamp = datetime.now()
        self.current_session["timestamps"].append(timestamp)
        
        for metric_name, metric_value in performance_metrics.items():
            if metric_name not in self.current_session["metrics"]:
                self.current_session["metrics"][metric_name] = []
            
            self.current_session["metrics"][metric_name].append(metric_value)
        
        # Log to MLflow if available
        if save_to_mlflow and self.mlflow_tracker:
            try:
                self.mlflow_tracker.log_metrics(performance_metrics, step=step)
                logger.debug(f"Logged {len(performance_metrics)} performance metrics to MLflow")
            except Exception as e:
                logger.error(f"Failed to log performance metrics to MLflow: {e}")
        
        logger.debug(f"Logged {len(performance_metrics)} performance metrics")
    
    def log_risk_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        step: Optional[int] = None,
        save_to_mlflow: bool = True
    ):
        """
        Log risk metrics for a series of returns.
        
        Args:
            returns: Series of returns
            benchmark_returns: Optional benchmark returns for comparison
            step: Step number for the metrics
            save_to_mlflow: Whether to save metrics to MLflow
        """
        if not self.current_session:
            logger.warning("No active session. Cannot log risk metrics.")
            return
        
        # Calculate risk metrics
        risk_metrics = self.risk_calculator.calculate_all_metrics(
            returns, benchmark_returns
        )
        
        # Store metrics
        timestamp = datetime.now()
        self.current_session["timestamps"].append(timestamp)
        
        for metric_name, metric_value in risk_metrics.items():
            if metric_name not in self.current_session["metrics"]:
                self.current_session["metrics"][metric_name] = []
            
            self.current_session["metrics"][metric_name].append(metric_value)
        
        # Log to MLflow if available
        if save_to_mlflow and self.mlflow_tracker:
            try:
                self.mlflow_tracker.log_metrics(risk_metrics, step=step)
                logger.debug(f"Logged {len(risk_metrics)} risk metrics to MLflow")
            except Exception as e:
                logger.error(f"Failed to log risk metrics to MLflow: {e}")
        
        logger.debug(f"Logged {len(risk_metrics)} risk metrics")
    
    def log_feature_importance(
        self,
        feature_names: List[str],
        importance_scores: List[float],
        method: str = "shap",
        step: Optional[int] = None,
        save_to_mlflow: bool = True
    ):
        """
        Log feature importance scores.
        
        Args:
            feature_names: List of feature names
            importance_scores: List of importance scores
            method: Method used to calculate importance (shap, permutation, etc.)
            step: Step number for the metrics
            save_to_mlflow: Whether to save metrics to MLflow
        """
        if not self.current_session:
            logger.warning("No active session. Cannot log feature importance.")
            return
        
        # Create feature importance metrics
        feature_metrics = {}
        for feature, importance in zip(feature_names, importance_scores):
            metric_name = f"feature_importance_{feature}"
            feature_metrics[metric_name] = float(importance)
        
        # Store metrics
        timestamp = datetime.now()
        self.current_session["timestamps"].append(timestamp)
        
        for metric_name, metric_value in feature_metrics.items():
            if metric_name not in self.current_session["metrics"]:
                self.current_session["metrics"][metric_name] = []
            
            self.current_session["metrics"][metric_name].append(metric_value)
        
        # Log to MLflow if available
        if save_to_mlflow and self.mlflow_tracker:
            try:
                self.mlflow_tracker.log_metrics(feature_metrics, step=step)
                logger.debug(f"Logged {len(feature_metrics)} feature importance metrics to MLflow")
            except Exception as e:
                logger.error(f"Failed to log feature importance to MLflow: {e}")
        
        logger.debug(f"Logged {len(feature_metrics)} feature importance metrics")
    
    def log_custom_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        save_to_mlflow: bool = True
    ):
        """
        Log custom metrics.
        
        Args:
            metrics: Dictionary of custom metrics
            step: Step number for the metrics
            save_to_mlflow: Whether to save metrics to MLflow
        """
        if not self.current_session:
            logger.warning("No active session. Cannot log custom metrics.")
            return
        
        # Store metrics
        timestamp = datetime.now()
        self.current_session["timestamps"].append(timestamp)
        
        for metric_name, metric_value in metrics.items():
            if metric_name not in self.current_session["metrics"]:
                self.current_session["metrics"][metric_name] = []
            
            self.current_session["metrics"][metric_name].append(metric_value)
        
        # Log to MLflow if available
        if save_to_mlflow and self.mlflow_tracker:
            try:
                self.mlflow_tracker.log_metrics(metrics, step=step)
                logger.debug(f"Logged {len(metrics)} custom metrics to MLflow")
            except Exception as e:
                logger.error(f"Failed to log custom metrics to MLflow: {e}")
        
        logger.debug(f"Logged {len(metrics)} custom metrics")
    
    def log_trade_metrics(
        self,
        trades: pd.DataFrame,
        step: Optional[int] = None,
        save_to_mlflow: bool = True
    ):
        """
        Log trade-related metrics.
        
        Args:
            trades: DataFrame containing trade information
            step: Step number for the metrics
            save_to_mlflow: Whether to save metrics to MLflow
        """
        if not self.current_session:
            logger.warning("No active session. Cannot log trade metrics.")
            return
        
        if trades.empty:
            logger.warning("No trades to analyze.")
            return
        
        # Calculate trade metrics
        trade_metrics = {}
        
        # Basic trade statistics
        trade_metrics["total_trades"] = len(trades)
        trade_metrics["winning_trades"] = len(trades[trades["pnl"] > 0])
        trade_metrics["losing_trades"] = len(trades[trades["pnl"] < 0])
        
        if len(trades) > 0:
            trade_metrics["win_rate"] = trade_metrics["winning_trades"] / trade_metrics["total_trades"]
            trade_metrics["avg_win"] = trades[trades["pnl"] > 0]["pnl"].mean() if trade_metrics["winning_trades"] > 0 else 0
            trade_metrics["avg_loss"] = trades[trades["pnl"] < 0]["pnl"].mean() if trade_metrics["losing_trades"] > 0 else 0
            trade_metrics["profit_factor"] = abs(trade_metrics["avg_win"] / trade_metrics["avg_loss"]) if trade_metrics["avg_loss"] != 0 else float('inf')
        
        # Store metrics
        timestamp = datetime.now()
        self.current_session["timestamps"].append(timestamp)
        
        for metric_name, metric_value in trade_metrics.items():
            if metric_name not in self.current_session["metrics"]:
                self.current_session["metrics"][metric_name] = []
            
            self.current_session["metrics"][metric_name].append(metric_value)
        
        # Log to MLflow if available
        if save_to_mlflow and self.mlflow_tracker:
            try:
                # Filter out infinite values for MLflow
                mlflow_metrics = {k: v for k, v in trade_metrics.items() if not np.isinf(v)}
                self.mlflow_tracker.log_metrics(mlflow_metrics, step=step)
                logger.debug(f"Logged {len(mlflow_metrics)} trade metrics to MLflow")
            except Exception as e:
                logger.error(f"Failed to log trade metrics to MLflow: {e}")
        
        logger.debug(f"Logged {len(trade_metrics)} trade metrics")
    
    def log_model_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        step: Optional[int] = None,
        save_to_mlflow: bool = True
    ):
        """
        Log model performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            step: Step number for the metrics
            save_to_mlflow: Whether to save metrics to MLflow
        """
        if not self.current_session:
            logger.warning("No active session. Cannot log model metrics.")
            return
        
        # Calculate model metrics
        model_metrics = {}
        
        # Classification metrics
        if len(np.unique(y_true)) == 2:  # Binary classification
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            model_metrics["accuracy"] = accuracy_score(y_true, y_pred)
            model_metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
            model_metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
            model_metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)
            
            if y_prob is not None:
                from sklearn.metrics import roc_auc_score
                try:
                    model_metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
                except ValueError:
                    pass
        
        # Regression metrics
        else:
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            model_metrics["mse"] = mean_squared_error(y_true, y_pred)
            model_metrics["rmse"] = np.sqrt(model_metrics["mse"])
            model_metrics["mae"] = mean_absolute_error(y_true, y_pred)
            model_metrics["r2_score"] = r2_score(y_true, y_pred)
        
        # Store metrics
        timestamp = datetime.now()
        self.current_session["timestamps"].append(timestamp)
        
        for metric_name, metric_value in model_metrics.items():
            if metric_name not in self.current_session["metrics"]:
                self.current_session["metrics"][metric_name] = []
            
            self.current_session["metrics"][metric_name].append(metric_value)
        
        # Log to MLflow if available
        if save_to_mlflow and self.mlflow_tracker:
            try:
                self.mlflow_tracker.log_metrics(model_metrics, step=step)
                logger.debug(f"Logged {len(model_metrics)} model metrics to MLflow")
            except Exception as e:
                logger.error(f"Failed to log model metrics to MLflow: {e}")
        
        logger.debug(f"Logged {len(model_metrics)} model metrics")
    
    def get_metrics_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of metrics for a session.
        
        Args:
            session_id: Session ID (defaults to current session)
            
        Returns:
            Dictionary containing metrics summary
        """
        session = self.current_session if session_id is None else self.metrics_history.get(session_id)
        
        if not session:
            return {}
        
        summary = {
            "session_id": session["id"],
            "session_name": session["name"],
            "start_time": session["start_time"].isoformat(),
            "end_time": session.get("end_time", "").isoformat(),
            "duration": session.get("duration", 0),
            "total_metrics": len(session["metrics"]),
            "total_steps": len(session["timestamps"])
        }
        
        # Calculate summary statistics for each metric
        metrics_summary = {}
        for metric_name, metric_values in session["metrics"].items():
            if metric_values:
                values = np.array(metric_values)
                metrics_summary[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "latest": float(values[-1]) if len(values) > 0 else None
                }
        
        summary["metrics_summary"] = metrics_summary
        
        return summary
    
    def export_metrics(
        self,
        session_id: Optional[str] = None,
        output_format: str = "csv"
    ) -> str:
        """
        Export metrics to a file.
        
        Args:
            session_id: Session ID (defaults to current session)
            output_format: Output format (csv, json, excel)
            
        Returns:
            Path to the exported file
        """
        session = self.current_session if session_id is None else self.metrics_history.get(session_id)
        
        if not session:
            raise ValueError("No session found for export")
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame(session["metrics"])
        metrics_df.index = session["timestamps"]
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_{session['name']}_{timestamp}"
        
        if output_format.lower() == "csv":
            output_file = os.path.join(session["output_dir"], f"{filename}.csv")
            metrics_df.to_csv(output_file)
        elif output_format.lower() == "json":
            output_file = os.path.join(session["output_dir"], f"{filename}.json")
            metrics_df.to_json(output_file, orient="index", indent=2)
        elif output_format.lower() == "excel":
            output_file = os.path.join(session["output_dir"], f"{filename}.xlsx")
            metrics_df.to_excel(output_file)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        logger.info(f"Exported metrics to: {output_file}")
        return output_file
    
    def _save_session_results(self):
        """Save current session results to file."""
        if not self.current_session:
            return
        
        # Save metrics data
        metrics_file = os.path.join(
            self.current_session["output_dir"],
            "metrics_data.json"
        )
        
        # Convert datetime objects to strings for JSON serialization
        session_data = self.current_session.copy()
        session_data["start_time"] = session_data["start_time"].isoformat()
        if session_data.get("end_time"):
            session_data["end_time"] = session_data["end_time"].isoformat()
        
        # Convert timestamps to strings
        session_data["timestamps"] = [ts.isoformat() for ts in session_data["timestamps"]]
        
        with open(metrics_file, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        logger.debug(f"Saved session results to: {metrics_file}")
    
    def cleanup_old_sessions(self, max_age_days: int = 30):
        """
        Clean up old metrics sessions.
        
        Args:
            max_age_days: Maximum age of sessions to keep
        """
        cutoff_date = datetime.now() - pd.Timedelta(days=max_age_days)
        
        sessions_to_remove = []
        for session_id, session in self.metrics_history.items():
            if session["start_time"] < cutoff_date:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            try:
                # Remove session directory
                import shutil
                shutil.rmtree(session["output_dir"], ignore_errors=True)
                
                # Remove from history
                del self.metrics_history[session_id]
                
                logger.info(f"Cleaned up old metrics session: {session_id}")
            except Exception as e:
                logger.error(f"Failed to cleanup session {session_id}: {e}")


def create_metrics_logger(
    output_dir: Optional[str] = None,
    auto_save: bool = True,
    mlflow_tracker: Optional[MLflowTracker] = None
) -> MetricsLogger:
    """
    Factory function to create metrics logger.
    
    Args:
        output_dir: Output directory for metrics
        auto_save: Whether to enable auto-save
        mlflow_tracker: Optional MLflow tracker
        
    Returns:
        Configured metrics logger instance
    """
    from ..core.config import get_config
    
    config = get_config()
    
    # Get configuration
    metrics_dir = output_dir or config.get('metrics', {}).get('output_dir', 'metrics_logs')
    auto_save_enabled = auto_save and config.get('metrics', {}).get('auto_save', True)
    
    return MetricsLogger(
        output_dir=metrics_dir,
        auto_save=auto_save_enabled,
        mlflow_tracker=mlflow_tracker
    )
