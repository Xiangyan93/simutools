#!/usr/bin/env python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod


class BaseForceField(ABC):
    @abstractmethod
    def check_version(self):
        pass

    @abstractmethod
    def checkout(self, **kwargs):
        pass
