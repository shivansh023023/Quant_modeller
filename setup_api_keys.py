#!/usr/bin/env python3
"""
API Key Setup Script for Quant Lab.

This script helps you set up your API keys for Quant Lab
Run this script to configure your Alpha Vantage and Gemini API keys.
"""

import os
import sys
from pathlib import Path
import yaml

def create_local_config():
    """Create a local configuration file with user input."""
    
    print("🚀 QUANT LAB API KEY SETUP")
    print("=" * 50)
    print()
    print("This script will help you set up your API keys for Quant Lab.")
    print("You'll need API keys for the following services:")
    print()
    print("📊 Alpha Vantage: Market data and fundamental data")
    print("🤖 Gemini AI: AI-powered strategy generation and research notes")
    print()
    
    # Check if local config already exists
    local_config_path = Path("configs/api_keys_local.yml")
    if local_config_path.exists():
        print("⚠️  Local configuration file already exists!")
        response = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if response != 'y':
            print("Setup cancelled.")
            return False
    
    # Get API keys from user
    print("\n🔑 Please enter your API keys:")
    print()
    
    # Alpha Vantage
    print("📊 ALPHA VANTAGE")
    print("Get your free API key at: https://www.alphavantage.co/support/#api-key")
    alpha_vantage_key = input("Alpha Vantage API Key: ").strip()
    
    if not alpha_vantage_key:
        print("⚠️  Alpha Vantage API key is required for market data functionality.")
        alpha_vantage_key = "YOUR_ALPHA_VANTAGE_API_KEY_HERE"
    
    # Gemini
    print("\n🤖 GEMINI AI")
    print("Get your free API key at: https://makersuite.google.com/app/apikey")
    gemini_key = input("Gemini API Key: ").strip()
    
    if not gemini_key:
        print("⚠️  Gemini API key is required for AI-powered features.")
        gemini_key = "YOUR_GEMINI_API_KEY_HERE"
    
    # Optional: Additional data sources
    print("\n🔍 OPTIONAL: Additional Data Sources")
    print("You can add these later if needed.")
    
    quandl_key = input("Quandl API Key (optional): ").strip() or None
    polygon_key = input("Polygon API Key (optional): ").strip() or None
    iex_key = input("IEX Cloud API Key (optional): ").strip() or None
    
    # Create configuration
    config = {
        'alpha_vantage': {
            'api_key': alpha_vantage_key,
            'base_url': 'https://www.alphavantage.co/query',
            'rate_limit': 5,
            'daily_limit': 500
        },
        'gemini': {
            'api_key': gemini_key,
            'base_url': 'https://generativelanguage.googleapis.com/v1beta',
            'model': 'gemini-pro',
            'rate_limit': 15,
            'max_tokens': 2048
        },
        'yfinance': {
            'enabled': True,
            'rate_limit': 2,
            'timeout': 10
        }
    }
    
    # Add optional services
    if quandl_key:
        config['quandl'] = {
            'api_key': quandl_key,
            'enabled': True
        }
    
    if polygon_key:
        config['polygon'] = {
            'api_key': polygon_key,
            'enabled': True
        }
    
    if iex_key:
        config['iex_cloud'] = {
            'api_key': iex_key,
            'enabled': True
        }
    
    # Ensure configs directory exists
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    # Write configuration file
    try:
        with open(local_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"\n✅ Configuration saved to: {local_config_path}")
        print()
        
        # Show summary
        print("📋 CONFIGURATION SUMMARY:")
        print(f"  Alpha Vantage: {'✓ Configured' if alpha_vantage_key != 'YOUR_ALPHA_VANTAGE_API_KEY_HERE' else '✗ Not configured'}")
        print(f"  Gemini AI: {'✓ Configured' if gemini_key != 'YOUR_GEMINI_API_KEY_HERE' else '✗ Not configured'}")
        print(f"  YFinance: ✓ Enabled (no API key needed)")
        
        if quandl_key:
            print(f"  Quandl: ✓ Configured")
        if polygon_key:
            print(f"  Polygon: ✓ Configured")
        if iex_key:
            print(f"  IEX Cloud: ✓ Configured")
        
        print()
        print("🔒 SECURITY NOTE:")
        print(f"  - Your API keys are stored in: {local_config_path}")
        print("  - This file is automatically ignored by git (.gitignore)")
        print("  - Never commit your actual API keys to version control!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        return False

def setup_environment_variables():
    """Set up environment variables for API keys."""
    
    print("\n🌍 ENVIRONMENT VARIABLES SETUP")
    print("=" * 50)
    print()
    print("You can also set API keys as environment variables.")
    print("This is useful for production deployments or CI/CD pipelines.")
    print()
    
    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if response != 'y':
            print("Environment variables setup skipped.")
            return False
    
    # Get API keys for environment variables
    print("🔑 Please enter your API keys for environment variables:")
    print()
    
    alpha_vantage_key = input("Alpha Vantage API Key: ").strip()
    gemini_key = input("Gemini API Key: ").strip()
    
    # Create .env file
    env_content = []
    
    if alpha_vantage_key:
        env_content.append(f"ALPHA_VANTAGE_API_KEY={alpha_vantage_key}")
    
    if gemini_key:
        env_content.append(f"GEMINI_API_KEY={gemini_key}")
    
    if env_content:
        try:
            with open(env_file, 'w') as f:
                f.write("\n".join(env_content))
                f.write("\n")
            
            print(f"\n✅ Environment variables saved to: {env_file}")
            print("🔒 This file is automatically ignored by git (.gitignore)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving environment variables: {e}")
            return False
    else:
        print("⚠️  No API keys provided. Environment variables setup skipped.")
        return False

def main():
    """Main setup function."""
    
    print("🚀 Welcome to Quant Lab API Key Setup!")
    print()
    
    # Check if we're in the right directory
    if not Path("src").exists() or not Path("configs").exists():
        print("❌ Error: Please run this script from the Quant Lab root directory.")
        print("   Make sure you're in the directory containing 'src' and 'configs' folders.")
        sys.exit(1)
    
    # Step 1: Create local configuration
    print("📝 STEP 1: Create Local Configuration File")
    print("-" * 40)
    
    if not create_local_config():
        print("❌ Failed to create local configuration. Setup incomplete.")
        sys.exit(1)
    
    # Step 2: Environment variables (optional)
    print("\n📝 STEP 2: Environment Variables (Optional)")
    print("-" * 40)
    
    response = input("Do you want to set up environment variables? (y/N): ").strip().lower()
    if response == 'y':
        setup_environment_variables()
    
    # Final instructions
    print("\n🎉 SETUP COMPLETE!")
    print("=" * 50)
    print()
    print("Your Quant Lab system is now configured with API keys.")
    print()
    print("📚 NEXT STEPS:")
    print("1. Test your configuration:")
    print("   python -c \"from src.core.config import config_manager; config_manager.print_config_summary()\"")
    print()
    print("2. Start using Quant Lab:")
    print("   python -c \"from src.core.config import get_api_key; print('Alpha Vantage:', '✓' if get_api_key('alpha_vantage') else '✗')\"")
    print()
    print("3. Check the README.md for usage examples")
    print()
    print("🔗 USEFUL LINKS:")
    print("  - Alpha Vantage: https://www.alphavantage.co/")
    print("  - Gemini AI: https://makersuite.google.com/")
    print("  - Quant Lab Docs: README.md")
    print()
    print("Happy trading! 🚀📈")

if __name__ == "__main__":
    main()
