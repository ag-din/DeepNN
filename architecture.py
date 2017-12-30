# -*- coding: utf-8 -*-
from copy import deepcopy


class Architecture:

    def __init__(self, id, type, create_model_function):
        self.id = id
        self.type = type
        self.create_model = create_model_function

    def __deepcopy__(self, memodict={}):
        return self

    def __repr__(self):
        return "arch:" + str(self.type) + "_" + str(self.id)
