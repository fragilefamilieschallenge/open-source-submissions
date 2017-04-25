'''
Created on Apr 2, 2017

@author: tsbertalan
'''
import time
import numpy as np
import pandas as pd
from contextlib import contextmanager


def thumb(badGood, unicode_symbols=True):
    symbols = ['{Disliked}', '{   Liked}']
    return symbols[badGood]

def wrap(txt, cols=69, indent=11):
    assert cols > indent
    if '\n' in txt:
        lines = txt.split('\n')
        output = []
        for line in lines:
            output.append(wrap(line, cols=cols, indent=indent))
        return '\n'.join(output)
    else:
        if len(txt) > cols:
            return txt[:cols] + '\n' + wrap(' '*indent + txt[cols:].strip(), cols=cols, indent=indent)
        else:
            return txt

def findClosest(vec, val):
    """Return the index where vec[index] is closest to val.
    >>> findClosest([2, 8, 3, 6], 5)
    3
    """
    distances = np.abs([val - x for x in vec])
    return distances.tolist().index(np.min(distances))

def _cachedResult(callback):
    '''Decorator to check for cached result.
    
    Some computations we might not want to do multiple times per RHS
    computation. Instead, we cache these until they're explictly invalidated
    (at the start of each RHS call).
    
    Use like
    >>> class SomeRHS(RHS):
    ...     param1 = 3 
    ...     param2 = 4
    ...     @_cachedResult
    ...     def computedValue(self):
    ...         print 'value = %s + %s' % (self.param1, self.param2)
    ...         return self.param1 + self.param2
    
    If we then instantiatem
    >>> r = SomeRHS()
    
    the property will execute the first time
    >>> r.computedValue()
    value = 3 + 4
    7
    
    but not the second.
    >>> r.computedValue()
    7
    
    Note that None is used for cache invalidation, so your methods cannot
    return None (or, they can, and will function correctly, but they won't get
    the benefit of any caching, since we'll assume the cache is empty each time).
    
    This also works with properties, but the @property decorator must come first.
    >>> class SomeRHS(RHS):
    ...     param1 = 3 
    ...     param2 = 4
    ...     @property
    ...     @_cachedResult
    ...     def computedValue(self):
    ...         print 'value = %s + %s' % (self.param1, self.param2)
    ...         return self.param1 + self.param2
    >>> r = SomeRHS()
    >>> r.computedValue
    value = 3 + 4
    7
    >>> r.computedValue
    7
    
    '''
    
    propertyName = callback.__name__
    def decoratedCallback(self, *args, **kwargs):
    
        if not hasattr(self, '_caches'):
            self._caches = {}
        # If there's a cached value, return it. If not, use the callback to
        # compute it, cache that result, and return it.
        if propertyName in self._caches:
            cachedValue = self._caches[propertyName]
        else:
            cachedValue = self._caches[propertyName] = None
        if cachedValue is None:
            cachedValue = callback(self, *args, **kwargs)
            self._caches[propertyName] = cachedValue
        
        # However obtained, return the (now cached) value.
        return cachedValue
    return decoratedCallback


def timeMethodDecorator(callback):
    
    methodName = callback.__name__
    def decoratedCallback(self, *args, **kwargs):
        start = time.time()
        output = callback(self, *args, **kwargs)
        decoratedName = '%sTime' % methodName
        setattr(self,decoratedName, time.time() - start)
        return output
    return decoratedCallback


class ListTable(list):
    """ Overridden list class which takes a 2-dimensional list of 
        the form [[1,2,3],[4,5,6]], and renders an HTML Table in 
        IPython Notebook.
        
        http://calebmadrigal.com/display-list-as-table-in-ipython-notebook/"""
    
    def _repr_html_(self):
        html = ["<table>"]
        for row in self:
            html.append("<tr>")
            
            for col in row:
                html.append("<td>{0}</td>".format(col))
            
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)
    
def getDefaultParams(obj):
    '''Find out what the default parameters would have been.'''
    defaultVersion = type(obj)()
    return defaultVersion.get_params()

def getNondefaultParams(obj):
    '''Find out what parameters have been changed from the defaults.'''
    out = {}
    if hasattr(obj, 'get_params'):
        try:
            defaults = getDefaultParams(obj)
            params = obj.get_params()
            for key in params:
                if params[key] != defaults[key]:
                    out[key] = params[key]
        except TypeError:
            pass
    return out

def describe(obj, addParams=True):
    out = obj.__class__.__name__.replace('__main__.', '').replace('Classifier', '')
    if addParams:
        if hasattr(obj, 'getNondefaultParams'):
            params = obj.getNondefaultParams()
        else:
            params = getNondefaultParams(obj)
        paramDescriptions = []
        for (k,v) in params.items():
            if callable(v):
                v = v.__name__
            paramDescriptions.append('%s=%s' % (k, v))
        params = ', '.join(paramDescriptions)
        if len(params) > 0:
            out += '(%s)' % (params,)
    return out

@contextmanager
def timeit(label=None):
    '''Context manager to print elapsed time.
    
    Use like:
    >>> with timeit('waiting'):
    ...     sleep(1.0)
    1.0 sec elapsed waiting.
    '''
    
    s = time.time()
    yield
    e = time.time()
    out = '%.1f sec elapsed' % (e - s)
    if label is not None:
        out += ' %s.' % label
    print out

def getDtype(s):
    return pd.DataFrame(['foo',],dtype=s)[0].dtype

def isCategory(dtype):
    category = getDtype('category')
    try:
        return dtype == category
    except TypeError:
        return False