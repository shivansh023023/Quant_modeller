"""
Feature registry for managing and organizing trading strategy features.

This module provides a centralized registry for features, allowing
easy lookup, validation, and management of feature definitions.
"""

import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import logging
from pathlib import Path

from ..strategies.schema import FeatureSpec

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureRegistry:
    """
    Centralized registry for managing trading strategy features.
    
    This class provides methods to register, lookup, validate, and
    manage features used in quantitative trading strategies.
    """
    
    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize the feature registry.
        
        Args:
            registry_path: Path to save/load registry (optional)
        """
        self.registry_path = registry_path or "data/features/feature_registry.json"
        self.features: Dict[str, FeatureSpec] = {}
        self.feature_functions: Dict[str, Callable] = {}
        self.metadata: Dict[str, Any] = {
            "created": datetime.now().isoformat(),
            "version": "1.0.0",
            "description": "Feature registry for Quant Lab trading strategies"
        }
        
        # Load existing registry if it exists
        self._load_registry()
    
    def register_feature(self, feature_spec: FeatureSpec, feature_func: Optional[Callable] = None) -> bool:
        """
        Register a new feature in the registry.
        
        Args:
            feature_spec: Feature specification
            feature_func: Function to compute the feature (optional)
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Validate feature specification
            if not self._validate_feature_spec(feature_spec):
                logger.error(f"Invalid feature specification for {feature_spec.name}")
                return False
            
            # Check for duplicates
            if feature_spec.name in self.features:
                logger.warning(f"Feature {feature_spec.name} already exists, updating...")
            
            # Register feature
            self.features[feature_spec.name] = feature_spec
            if feature_func:
                self.feature_functions[feature_spec.name] = feature_func
            
            logger.info(f"Registered feature: {feature_spec.name}")
            
            # Save registry
            self._save_registry()
            return True
            
        except Exception as e:
            logger.error(f"Failed to register feature {feature_spec.name}: {e}")
            return False
    
    def get_feature(self, name: str) -> Optional[FeatureSpec]:
        """
        Get a feature specification by name.
        
        Args:
            name: Feature name
            
        Returns:
            Feature specification or None if not found
        """
        return self.features.get(name)
    
    def get_feature_function(self, name: str) -> Optional[Callable]:
        """
        Get a feature computation function by name.
        
        Args:
            name: Feature name
            
        Returns:
            Feature function or None if not found
        """
        return self.feature_functions.get(name)
    
    def list_features(self, feature_type: Optional[str] = None) -> List[FeatureSpec]:
        """
        List all features, optionally filtered by type.
        
        Args:
            feature_type: Filter by feature type (optional)
            
        Returns:
            List of feature specifications
        """
        if feature_type:
            return [
                feature for feature in self.features.values()
                if feature.feature_type.value == feature_type
            ]
        return list(self.features.values())
    
    def search_features(self, query: str) -> List[FeatureSpec]:
        """
        Search features by name.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching feature specifications
        """
        query_lower = query.lower()
        matches = []
        
        for feature in self.features.values():
            if query_lower in feature.name.lower():
                matches.append(feature)
        
        return matches
    
    def remove_feature(self, name: str) -> bool:
        """
        Remove a feature from the registry.
        
        Args:
            name: Feature name to remove
            
        Returns:
            True if removal successful, False otherwise
        """
        try:
            if name in self.features:
                del self.features[name]
                if name in self.feature_functions:
                    del self.feature_functions[name]
                
                logger.info(f"Removed feature: {name}")
                self._save_registry()
                return True
            else:
                logger.warning(f"Feature {name} not found in registry")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove feature {name}: {e}")
            return False
    
    def update_feature(self, name: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing feature specification.
        
        Args:
            name: Feature name to update
            updates: Dictionary of updates to apply
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            if name not in self.features:
                logger.error(f"Feature {name} not found in registry")
                return False
            
            feature = self.features[name]
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(feature, key):
                    setattr(feature, key, value)
                else:
                    logger.warning(f"Invalid field {key} for feature {name}")
            
            # Validate updated feature
            if not self._validate_feature_spec(feature):
                logger.error(f"Updated feature {name} is invalid")
                return False
            
            logger.info(f"Updated feature: {name}")
            self._save_registry()
            return True
            
        except Exception as e:
            logger.error(f"Failed to update feature {name}: {e}")
            return False
    
    def get_feature_dependencies(self, name: str) -> List[str]:
        """
        Get dependencies for a feature.
        
        Args:
            name: Feature name
            
        Returns:
            List of dependency names
        """
        feature = self.get_feature(name)
        if not feature:
            return []
        
        dependencies = []
        
        # Check for data source dependencies
        if hasattr(feature, 'data_source') and feature.data_source:
            dependencies.append(f"data_source:{feature.data_source}")
        
        # Check for feature dependencies (if feature uses other features)
        if hasattr(feature, 'depends_on') and feature.depends_on:
            dependencies.extend(feature.depends_on)
        
        return dependencies
    
    def validate_feature_consistency(self) -> Dict[str, List[str]]:
        """
        Validate consistency across all features.
        
        Returns:
            Dictionary of validation issues by category
        """
        issues = {
            "missing_dependencies": [],
            "circular_dependencies": [],
            "invalid_lookbacks": [],
            "naming_conflicts": []
        }
        
        # Check for missing dependencies
        for name, feature in self.features.items():
            dependencies = self.get_feature_dependencies(name)
            for dep in dependencies:
                if dep.startswith("feature:") and dep[8:] not in self.features:
                    issues["missing_dependencies"].append(f"{name} -> {dep}")
        
        # Check for invalid lookback periods
        for name, feature in self.features.items():
            if feature.lookback_period <= 0:
                issues["invalid_lookbacks"].append(f"{name}: lookback_period <= 0")
            elif feature.lookback_period > 252 * 5:  # 5 years max
                issues["invalid_lookbacks"].append(f"{name}: lookback_period > 5 years")
        
        # Check for naming conflicts (similar names)
        names = list(self.features.keys())
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                if self._names_similar(name1, name2):
                    issues["naming_conflicts"].append(f"{name1} vs {name2}")
        
        return issues
    
    def export_registry(self, export_path: str) -> bool:
        """
        Export the feature registry to a file.
        
        Args:
            export_path: Path to export the registry
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            export_data = {
                "metadata": self.metadata,
                "features": {name: feature.dict() for name, feature in self.features.items()},
                "exported_at": datetime.now().isoformat()
            }
            
            # Ensure directory exists
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported feature registry to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export registry: {e}")
            return False
    
    def import_registry(self, import_path: str, merge: bool = True) -> bool:
        """
        Import a feature registry from a file.
        
        Args:
            import_path: Path to import the registry from
            merge: Whether to merge with existing registry (True) or replace (False)
            
        Returns:
            True if import successful, False otherwise
        """
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            if not merge:
                # Replace existing registry
                self.features.clear()
                self.feature_functions.clear()
                self.metadata = import_data.get("metadata", {})
            
            # Import features
            imported_count = 0
            for name, feature_data in import_data.get("features", {}).items():
                try:
                    feature_spec = FeatureSpec(**feature_data)
                    if name not in self.features or not merge:
                        self.features[name] = feature_spec
                        imported_count += 1
                except Exception as e:
                    logger.warning(f"Failed to import feature {name}: {e}")
            
            logger.info(f"Imported {imported_count} features from {import_path}")
            self._save_registry()
            return True
            
        except Exception as e:
            logger.error(f"Failed to import registry: {e}")
            return False
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the feature registry.
        
        Returns:
            Dictionary of registry statistics
        """
        feature_types = {}
        for feature in self.features.values():
            feature_type = feature.feature_type.value
            feature_types[feature_type] = feature_types.get(feature_type, 0) + 1
        
        return {
            "total_features": len(self.features),
            "feature_types": feature_types,
            "features_with_functions": len(self.feature_functions),
            "registry_size": len(json.dumps(self.features)),
            "last_updated": self.metadata.get("last_updated", "unknown")
        }
    
    def _validate_feature_spec(self, feature_spec: FeatureSpec) -> bool:
        """
        Validate a feature specification.
        
        Args:
            feature_spec: Feature specification to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic validation
            if not feature_spec.name or not feature_spec.name.strip():
                return False
            
            if feature_spec.lookback_period <= 0:
                return False
            
            # FeatureSpec doesn't have a description field, skip this check
            
            # Check for reserved names
            reserved_names = {"open", "high", "low", "close", "volume", "returns"}
            if feature_spec.name.lower() in reserved_names:
                logger.warning(f"Feature name {feature_spec.name} is reserved")
            
            return True
            
        except Exception as e:
            logger.error(f"Feature validation error: {e}")
            return False
    
    def _names_similar(self, name1: str, name2: str, threshold: float = 0.8) -> bool:
        """
        Check if two feature names are similar.
        
        Args:
            name1: First feature name
            name2: Second feature name
            threshold: Similarity threshold
            
        Returns:
            True if names are similar
        """
        # Simple similarity check - can be enhanced with more sophisticated algorithms
        name1_lower = name1.lower()
        name2_lower = name2.lower()
        
        # Check if one name contains the other
        if name1_lower in name2_lower or name2_lower in name1_lower:
            return True
        
        # Check for common prefixes/suffixes
        if (name1_lower.startswith(name2_lower[:3]) or 
            name1_lower.endswith(name2_lower[-3:])):
            return True
        
        return False
    
    def _save_registry(self) -> None:
        """Save the registry to disk."""
        try:
            # Ensure directory exists
            Path(self.registry_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Update metadata
            self.metadata["last_updated"] = datetime.now().isoformat()
            self.metadata["feature_count"] = len(self.features)
            
            # Save registry
            registry_data = {
                "metadata": self.metadata,
                "features": {name: feature.dict() for name, feature in self.features.items()}
            }
            
            with open(self.registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def _load_registry(self) -> None:
        """Load the registry from disk."""
        try:
            if Path(self.registry_path).exists():
                with open(self.registry_path, 'r') as f:
                    registry_data = json.load(f)
                
                # Load metadata
                self.metadata = registry_data.get("metadata", self.metadata)
                
                # Load features
                for name, feature_data in registry_data.get("features", {}).items():
                    try:
                        feature_spec = FeatureSpec(**feature_data)
                        self.features[name] = feature_spec
                    except Exception as e:
                        logger.warning(f"Failed to load feature {name}: {e}")
                
                logger.info(f"Loaded {len(self.features)} features from registry")
                
        except Exception as e:
            logger.warning(f"Failed to load registry: {e}")
            # Continue with empty registry
    
    def has_feature(self, name: str) -> bool:
        """
        Check if a feature exists in the registry.
        
        Args:
            name: Feature name to check
            
        Returns:
            True if feature exists, False otherwise
        """
        return name in self.features
    
    def validate_feature_spec(self, feature_spec: FeatureSpec) -> bool:
        """
        Validate a feature specification.
        
        Args:
            feature_spec: Feature specification to validate
            
        Returns:
            True if valid, False otherwise
        """
        return self._validate_feature_spec(feature_spec)
    
    def get_available_categories(self) -> List[str]:
        """
        Get list of available feature categories.
        
        Returns:
            List of category names
        """
        from .generators import FeatureCategory
        return [cat.value for cat in FeatureCategory]
    
    def register_feature_with_function(
        self,
        name: str,
        feature_function: Callable,
        feature_type: Any,
        description: str = "",
        lookback_period: int = 20,
        parameters: Dict[str, Any] = None
    ) -> bool:
        """
        Register a feature with function (alternative signature for tests).
        
        Args:
            name: Feature name
            feature_function: Function to compute the feature
            feature_type: Type of feature
            description: Feature description
            lookback_period: Lookback period
            parameters: Feature parameters
            
        Returns:
            True if registration successful
        """
        feature_spec = FeatureSpec(
            name=name,
            feature_type=feature_type,
            lookback_period=lookback_period,
            parameters=parameters or {}
        )
        
        return self.register_feature(feature_spec, feature_function)
