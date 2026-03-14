#!/usr/bin/env python3
"""
Test script for Quant Lab configuration system

This script tests the configuration manager and API key handling.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_config_manager():
    """Test the configuration manager."""
    print("🧪 Testing Configuration Manager...")
    
    try:
        from src.core.config import config_manager
        
        print("✅ Configuration manager imported successfully")
        
        # Test configuration summary
        print("\n📋 Configuration Summary:")
        config_manager.print_config_summary()
        
        # Test API key retrieval
        print("\n🔑 API Key Status:")
        alpha_vantage_key = config_manager.get_api_key('alpha_vantage')
        gemini_key = config_manager.get_api_key('gemini')
        
        print(f"  Alpha Vantage: {'✓' if alpha_vantage_key and alpha_vantage_key != 'YOUR_ALPHA_VANTAGE_API_KEY_HERE' else '✗'}")
        print(f"  Gemini AI: {'✓' if gemini_key and gemini_key != 'YOUR_GEMINI_API_KEY_HERE' else '✗'}")
        
        # Test service status
        print("\n🔧 Service Status:")
        print(f"  Alpha Vantage enabled: {config_manager.is_service_enabled('alpha_vantage')}")
        print(f"  Gemini enabled: {config_manager.is_service_enabled('gemini')}")
        print(f"  YFinance enabled: {config_manager.is_service_enabled('yfinance')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing configuration manager: {e}")
        return False

def test_imports():
    """Test basic imports."""
    print("🧪 Testing Basic Imports...")
    
    try:
        # Test core imports
        from src.core.data_api import DataAPI, DataCatalog
        print("✅ Core modules imported successfully")
        
        # Test strategies imports
        from src.strategies.schema import StrategySpec
        print("✅ Strategy modules imported successfully")
        
        # Test features imports
        from src.features.registry import FeatureRegistry
        print("✅ Feature modules imported successfully")
        
        # Test models imports
        from src.models.model_zoo import ModelZoo
        print("✅ Model modules imported successfully")
        
        # Test AI imports
        from src.ai.idea_generator import IdeaGenerator
        print("✅ AI modules imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing imports: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 QUANT LAB CONFIGURATION TEST")
    print("=" * 50)
    
    # Test 1: Basic imports
    if not test_imports():
        print("\n❌ Basic imports failed. Please check your installation.")
        return False
    
    # Test 2: Configuration manager
    if not test_config_manager():
        print("\n❌ Configuration manager test failed.")
        return False
    
    print("\n🎉 ALL TESTS PASSED!")
    print("=" * 50)
    print("\nYour Quant Lab system is properly configured and ready to use!")
    print("\n📚 Next steps:")
    print("1. If you see ✗ for API keys, run: python setup_api_keys.py")
    print("2. Start building strategies with the AI-powered tools")
    print("3. Check the README.md for usage examples")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
