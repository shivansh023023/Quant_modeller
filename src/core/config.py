"""
Configuration management for Quant Lab.

This module provides secure configuration management for API keys,
settings, and environment variables.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration and API keys securely.
    
    This class handles loading configuration from multiple sources
    with proper security practices.
    """
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config = {}
        self.api_keys = {}
        
        # Load environment variables
        load_dotenv()
        
        # Load configuration files
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from multiple sources."""
        # Priority order: environment variables > local config > template config
        
        # 1. Load template configuration
        template_config = self._load_yaml_file("api_keys.yml")
        if template_config:
            self.config.update(template_config)
        
        # 2. Load local configuration (overrides template)
        local_config = self._load_yaml_file("api_keys_local.yml")
        if local_config:
            self.config.update(local_config)
        
        # 3. Load environment variables (highest priority)
        self._load_from_env()
        
        # 4. Extract API keys
        self._extract_api_keys()
        
        # 5. Validate configuration
        self._validate_config()
    
    def _load_yaml_file(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load a YAML configuration file."""
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Configuration file not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from: {filepath}")
                return config
        except Exception as e:
            logger.error(f"Failed to load configuration from {filepath}: {e}")
            return None
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Alpha Vantage
        if os.getenv('ALPHA_VANTAGE_API_KEY'):
            if 'alpha_vantage' not in self.config:
                self.config['alpha_vantage'] = {}
            self.config['alpha_vantage']['api_key'] = os.getenv('ALPHA_VANTAGE_API_KEY')
        
        # Gemini
        if os.getenv('GEMINI_API_KEY'):
            if 'gemini' not in self.config:
                self.config['gemini'] = {}
            self.config['gemini']['api_key'] = os.getenv('GEMINI_API_KEY')
        
        # Other data sources
        if os.getenv('QUANDL_API_KEY'):
            if 'quandl' not in self.config:
                self.config['quandl'] = {}
            self.config['quandl']['api_key'] = os.getenv('QUANDL_API_KEY')
        
        if os.getenv('POLYGON_API_KEY'):
            if 'polygon' not in self.config:
                self.config['polygon'] = {}
            self.config['polygon']['api_key'] = os.getenv('POLYGON_API_KEY')
        
        if os.getenv('IEX_CLOUD_API_KEY'):
            if 'iex_cloud' not in self.config:
                self.config['iex_cloud'] = {}
            self.config['iex_cloud']['api_key'] = os.getenv('IEX_CLOUD_API_KEY')
    
    def _extract_api_keys(self) -> None:
        """Extract API keys from configuration."""
        # Alpha Vantage
        if 'alpha_vantage' in self.config:
            self.api_keys['alpha_vantage'] = self.config['alpha_vantage'].get('api_key')
        
        # Gemini
        if 'gemini' in self.config:
            self.api_keys['gemini'] = self.config['gemini'].get('api_key')
        
        # Other sources
        for source in ['quandl', 'polygon', 'iex_cloud']:
            if source in self.config:
                self.api_keys[source] = self.config[source].get('api_key')
    
    def _validate_config(self) -> None:
        """Validate the configuration."""
        warnings = []
        
        # Check for required API keys
        if not self.api_keys.get('alpha_vantage'):
            warnings.append("Alpha Vantage API key not found - market data functionality will be limited")
        
        if not self.api_keys.get('gemini'):
            warnings.append("Gemini API key not found - AI-powered features will be disabled")
        
        # Check for configuration issues
        if 'alpha_vantage' in self.config:
            av_config = self.config['alpha_vantage']
            if av_config.get('api_key') == "YOUR_ALPHA_VANTAGE_API_KEY_HERE":
                warnings.append("Alpha Vantage API key appears to be template value")
        
        if 'gemini' in self.config:
            gemini_config = self.config['gemini']
            if gemini_config.get('api_key') == "YOUR_GEMINI_API_KEY_HERE":
                warnings.append("Gemini API key appears to be template value")
        
        # Log warnings
        for warning in warnings:
            logger.warning(warning)
    
    def get_api_key(self, service: str) -> Optional[str]:
        """
        Get API key for a specific service.
        
        Args:
            service: Service name (e.g., 'alpha_vantage', 'gemini')
            
        Returns:
            API key or None if not found
        """
        return self.api_keys.get(service)
    
    def get_config(self, section: str, key: str = None, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key (optional)
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        if section not in self.config:
            return default
        
        if key is None:
            return self.config[section]
        
        return self.config[section].get(key, default)
    
    def has_api_key(self, service: str) -> bool:
        """
        Check if API key exists for a service.
        
        Args:
            service: Service name
            
        Returns:
            True if API key exists, False otherwise
        """
        return self.get_api_key(service) is not None
    
    def get_service_config(self, service: str) -> Dict[str, Any]:
        """
        Get full configuration for a service.
        
        Args:
            service: Service name
            
        Returns:
            Service configuration dictionary
        """
        return self.config.get(service, {})
    
    def is_service_enabled(self, service: str) -> bool:
        """
        Check if a service is enabled.
        
        Args:
            service: Service name
            
        Returns:
            True if service is enabled, False otherwise
        """
        service_config = self.get_service_config(service)
        return service_config.get('enabled', True)
    
    def get_rate_limits(self, service: str) -> Dict[str, int]:
        """
        Get rate limits for a service.
        
        Args:
            service: Service name
            
        Returns:
            Dictionary of rate limit information
        """
        service_config = self.get_service_config(service)
        return {
            'rate_limit': service_config.get('rate_limit', 0),
            'daily_limit': service_config.get('daily_limit', 0)
        }
    
    def create_local_config_template(self) -> bool:
        """
        Create a local configuration template from the main template.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            template_path = self.config_dir / "api_keys.yml"
            local_path = self.config_dir / "api_keys_local.yml"
            
            if not template_path.exists():
                logger.error("Template configuration file not found")
                return False
            
            if local_path.exists():
                logger.warning("Local configuration file already exists")
                return False
            
            # Copy template to local
            with open(template_path, 'r') as f:
                template_content = f.read()
            
            with open(local_path, 'w') as f:
                f.write(template_content)
            
            logger.info(f"Created local configuration template: {local_path}")
            logger.info("Please edit this file with your actual API keys")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create local config template: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current configuration.
        
        Returns:
            Dictionary containing configuration summary
        """
        summary = {
            'config_dir': str(self.config_dir),
            'services_configured': list(self.config.keys()),
            'api_keys_available': list(self.api_keys.keys()),
            'services_enabled': {},
            'warnings': []
        }
        
        # Check service status
        for service in self.config.keys():
            summary['services_enabled'][service] = self.is_service_enabled(service)
        
        # Check for missing API keys
        if not self.api_keys.get('alpha_vantage'):
            summary['warnings'].append("Alpha Vantage API key missing")
        
        if not self.api_keys.get('gemini'):
            summary['warnings'].append("Gemini API key missing")
        
        return summary
    
    def print_config_summary(self) -> None:
        """Print a human-readable configuration summary."""
        summary = self.get_config_summary()
        
        print("\n" + "="*50)
        print("QUANT LAB CONFIGURATION SUMMARY")
        print("="*50)
        
        print(f"\nConfiguration Directory: {summary['config_dir']}")
        
        print(f"\nServices Configured: {len(summary['services_configured'])}")
        for service in summary['services_configured']:
            status = "✓ ENABLED" if summary['services_enabled'][service] else "✗ DISABLED"
            has_key = "✓ KEY" if self.has_api_key(service) else "✗ NO KEY"
            print(f"  {service}: {status} | {has_key}")
        
        if summary['warnings']:
            print(f"\n⚠️  WARNINGS:")
            for warning in summary['warnings']:
                print(f"  - {warning}")
        
        print("\n" + "="*50)


# Global configuration instance
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    return config_manager


def get_api_key(service: str) -> Optional[str]:
    """Get API key for a service (convenience function)."""
    return config_manager.get_api_key(service)


def is_service_enabled(service: str) -> bool:
    """Check if a service is enabled (convenience function)."""
    return config_manager.is_service_enabled(service)
