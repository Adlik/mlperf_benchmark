"""
TVM
"""

# pylint: disable=unused-argument,missing-docstring,useless-super-delegation

from threading import Lock
import os
import time
import numpy as np

import backend
import tvm
from tvm.contrib import graph_runtime


class BackendTVM(backend.Backend):
    def __init__(self):
        super(BackendTVM, self).__init__()

    def version(self):
        try:
            return tvm.__version__
        except AttributeError:
            return 0

    def name(self):
        return "tvm"

    def image_format(self):
        return "NCHW"

    def load(self, model_path, inputs=None, outputs=None):
        ctx = tvm.cpu()
        lib = tvm.runtime.load_module(model_path)
        self.module = graph_runtime.GraphModule(lib["default"](ctx))
        if inputs:
            self.inputs = inputs
        else:
            raise Exception('Input name(s) of model is required when using TVM backend')
        return self

    def predict(self, feed):
        self.module.set_input(self.inputs[0], feed[self.inputs[0]])
        self.module.run()
        out = self.module.get_output(0).asnumpy()
        return [out]
