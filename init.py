import os.path
import sys


caffe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'caffe', 'python')
if caffe_path not in sys.path:
    sys.path.insert(0, caffe_path)
