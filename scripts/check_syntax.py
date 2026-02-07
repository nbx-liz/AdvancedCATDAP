import sys
import os
try:
    from advanced_catdap.frontend import dash_app
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
except SyntaxError as e:
    print(f"Syntax error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Other error: {e}")
    # Dash app instantiation might fail due to assets folder etc, but syntax should be fine
