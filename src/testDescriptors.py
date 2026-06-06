import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from core.bandDescriptors import format_descriptors_for_prompt
 
print(format_descriptors_for_prompt(6))