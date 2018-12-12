"""
Copyright (c) 2009-2016 by California Institute of Technology
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder(s) nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
COPYRIGHT HOLDERS OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.

This module is modified from python-control.
Reference: https://github.com/python-control/python-control
"""
import warnings


__all__ = ['LinearTimeInvariant']


class LinearTimeInvariant(object):
    """
    This class is a parent class, which implements linear
    time invariant system (LTI).

    :param inputs: the number of input channels
    :type inputs: int
    :param outputs: the number of output channels
    :type outputs: int
    :param dt: sampling time. dt is None or 0 which means the system is a
           continuous system. dt > 0 represents a discrete system. And dt < 0
           is invalid.
    :type dt: int | float
    """
    def __init__(self, inputs=1, outputs=1, dt=None):
        self.inputs = inputs
        self.outputs = outputs
        self.dt = dt

    def isctime(self, strict=False):
        warnings.warn('isctime method is deprecated, use is_ctime property instead',
                      DeprecationWarning)
        if self.dt is None:
            return False if strict else True
        return self.dt == 0

    def isdtime(self, strict):
        warnings.warn('isdtime method is deprecated, use is_dtime property instead',
                      DeprecationWarning)
        if self.dt is None:
            return False if strict else True
        return self.dt > 0

    def issiso(self):
        warnings.warn('issiso method is deprecated, use is_siso property instead',
                      DeprecationWarning)
        return self.outputs == 1 and self.inputs == 1

    @property
    def is_ctime(self):
        return self.dt is None or self.dt == 0

    @property
    def is_dtime(self):
        return self.dt is not None and self.dt > 0

    @property
    def is_siso(self):
        return self.outputs == 1 and self.inputs == 1

    @property
    def is_gain(self):
        raise NotImplementedError('This should be implemented by subclass')


def isctime(sys_, strict=False):
    if sys_.dt is None:
        return False if strict else True
    return sys_.dt == 0


def isdtime(sys_, strict=False):
    if sys_.dt is None:
        return False if strict else True
    return sys_.dt > 0


def issiso(sys_):
    return sys_.outputs == 1 and sys_.inputs == 1
