#!/usr/bin/env python3
"""
Sportka Predictor - Launcher Script

This script provides a simple command-line interface to launch
different components of the Sportka Predictor system.
"""

import sys
import argparse


def launch_gui():
    """Launch the graphical user interface."""
    from sportka.gui import main
    print("Launching Sportka Predictor GUI...")
    main()


def launch_cli():
    """Launch the command-line predictor."""
    from sportka.learn_improved import main
    return main()


def create_config():
    """Create default configuration file."""
    from sportka.config import create_default_config
    create_default_config()


def test_installation():
    """Test if all dependencies are installed correctly."""
    print("Testing Sportka Predictor installation...")
    print()
    
    errors = []
    warnings = []
    
    # Test imports
    modules = [
        ('numpy', 'NumPy', False),
        ('tensorflow', 'TensorFlow', False),
        ('tkinter', 'Tkinter', True),  # Optional for GUI
        ('reportlab', 'ReportLab', False),
        ('ephem', 'PyEphem', False),
    ]
    
    for module_name, display_name, optional in modules:
        try:
            __import__(module_name)
            print(f"✓ {display_name} installed")
        except ImportError:
            if optional:
                print(f"⚠ {display_name} NOT installed (optional - GUI will not work)")
                warnings.append(display_name)
            else:
                print(f"✗ {display_name} NOT installed")
                errors.append(display_name)
    
    # Test our modules
    print()
    our_modules = [
        ('sportka.biquaternion', 'Biquaternion module'),
        ('sportka.neural_network', 'Neural network module'),
        ('sportka.gui', 'GUI module'),
        ('sportka.config', 'Configuration module'),
    ]
    
    for module_name, display_name in our_modules:
        try:
            __import__(module_name)
            print(f"✓ {display_name} loaded")
        except Exception as e:
            print(f"✗ {display_name} FAILED: {e}")
            errors.append(display_name)
    
    print()
    if errors:
        print(f"❌ Installation incomplete. Missing: {', '.join(errors)}")
        return 1
    elif warnings:
        print(f"⚠️  Installation complete with warnings: {', '.join(warnings)}")
        print("   Core functionality available. GUI requires tkinter.")
        return 0
    else:
        print("✅ All components installed correctly!")
        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Sportka Predictor - Advanced ML Lottery System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s gui          Launch the graphical interface
  %(prog)s predict      Run prediction with current settings
  %(prog)s config       Create default configuration file
  %(prog)s test         Test installation

For more information, see sportka/README.md
        """
    )
    
    parser.add_argument(
        'command',
        choices=['gui', 'predict', 'config', 'test'],
        help='Command to run'
    )
    
    args = parser.parse_args()
    
    if args.command == 'gui':
        launch_gui()
    elif args.command == 'predict':
        return launch_cli()
    elif args.command == 'config':
        create_config()
    elif args.command == 'test':
        return test_installation()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
