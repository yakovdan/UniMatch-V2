import os
import sys

# The training scripts live at the repo root and import their helpers as top-level
# packages (util, model, dataset), so the tests need the root on sys.path whether
# they are launched from the root or from tests/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
