"""
Configuration management for Sportka Predictor.

Handles user preferences and system settings.
"""

import json
import os
from typing import Dict, Any


DEFAULT_CONFIG = {
    "data": {
        "default_csv_path": "/tmp/sportka.csv",
        "auto_download": False,
        "download_on_startup": False
    },
    "training": {
        "default_epochs": 100,
        "default_batch_size": 32,
        "default_hidden_layers": 32,
        "default_hidden_units": 128,
        "default_dropout_rate": 0.3,
        "use_biquaternion": True,
        "validation_split": 0.2
    },
    "prediction": {
        "number_of_predictions": 7,
        "generate_alternatives": 3,
        "alternative_pool_size": 12
    },
    "model": {
        "model_dir": "./models",
        "auto_save": True,
        "load_latest_on_startup": False
    },
    "gui": {
        "window_width": 1000,
        "window_height": 800,
        "theme": "default",
        "font_size": 10
    },
    "output": {
        "pdf_output_dir": "./predictions",
        "json_output_dir": "./predictions",
        "date_format": "%d.%m.%Y"
    }
}


class Config:
    """Configuration manager."""
    
    def __init__(self, config_path: str = "./config.json"):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = DEFAULT_CONFIG.copy()
        self.load()
    
    def load(self) -> None:
        """Load configuration from file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = json.load(f)
                
                # Merge with defaults
                self._merge_config(user_config)
                
                print(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                print(f"Warning: Could not load config: {e}")
                print("Using default configuration")
    
    def save(self) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"Configuration saved to {self.config_path}")
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def _merge_config(self, user_config: Dict[str, Any]) -> None:
        """Merge user configuration with defaults."""
        for section, values in user_config.items():
            if section in self.config:
                if isinstance(values, dict):
                    self.config[section].update(values)
                else:
                    self.config[section] = values
            else:
                self.config[section] = values
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            default: Default value if not found
        
        Returns:
            Configuration value
        """
        return self.config.get(section, {}).get(key, default)
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Value to set
        """
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Configuration section name
        
        Returns:
            Configuration section dictionary
        """
        return self.config.get(section, {})
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self.config = DEFAULT_CONFIG.copy()
    
    def export_config(self, path: str) -> None:
        """
        Export configuration to a specific file.
        
        Args:
            path: Export file path
        """
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Configuration exported to {path}")
    
    def import_config(self, path: str) -> None:
        """
        Import configuration from a file.
        
        Args:
            path: Import file path
        """
        with open(path, 'r') as f:
            imported_config = json.load(f)
        
        self._merge_config(imported_config)
        print(f"Configuration imported from {path}")


def create_default_config(path: str = "./config.json") -> None:
    """
    Create a default configuration file.
    
    Args:
        path: Path to create configuration file
    """
    if os.path.exists(path):
        response = input(f"{path} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Configuration creation cancelled")
            return
    
    with open(path, 'w') as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    
    print(f"Default configuration created at {path}")


if __name__ == '__main__':
    # Create default config when run as script
    import sys
    
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "./config.json"
    
    create_default_config(config_path)
