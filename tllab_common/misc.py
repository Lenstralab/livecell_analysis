import os
import re
import yaml
from copy import deepcopy


loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))


class color_class(object):
    """ print colored text:
            print(color('Hello World!', 'r:b'))
            print(color % 'r:b' + 'Hello World! + color)
            print(f'{color("r:b")}Hello World!{color}')
        text: text to be colored/decorated
        fmt: string: 'k': black, 'r': red', 'g': green, 'y': yellow, 'b': blue, 'm': magenta, 'c': cyan, 'w': white
            'b'  text color
            '.r' background color
            ':b' decoration: 'b': bold, 'u': underline, 'r': reverse
            for colors also terminal color codes can be used

        example: >> print(color('Hello World!', 'b.208:b'))
                 << Hello world! in blue bold on orange background

        wp@tl20191122
    """

    def __init__(self, fmt=None):
        self._open = False

    def _fmt(self, fmt=None):
        if fmt is None:
            self._open = False
            return '\033[0m'

        if not isinstance(fmt, str):
            fmt = str(fmt)

        decorS = [i.group(0) for i in re.finditer('(?<=\:)[a-zA-Z]', fmt)]
        backcS = [i.group(0) for i in re.finditer('(?<=\.)[a-zA-Z]', fmt)]
        textcS = [i.group(0) for i in re.finditer('((?<=[^\.\:])|^)[a-zA-Z]', fmt)]
        backcN = [i.group(0) for i in re.finditer('(?<=\.)\d{1,3}', fmt)]
        textcN = [i.group(0) for i in re.finditer('((?<=[^\.\:\d])|^)\d{1,3}', fmt)]

        t = 'krgybmcw'
        d = {'b': 1, 'u': 4, 'r': 7}

        text = ''
        for i in decorS:
            if i.lower() in d:
                text = '\033[{}m{}'.format(d[i.lower()], text)
        for i in backcS:
            if i.lower() in t:
                text = '\033[48;5;{}m{}'.format(t.index(i.lower()), text)
        for i in textcS:
            if i.lower() in t:
                text = '\033[38;5;{}m{}'.format(t.index(i.lower()), text)
        for i in backcN:
            if 0 <= int(i) <= 255:
                text = '\033[48;5;{}m{}'.format(int(i), text)
        for i in textcN:
            if 0 <= int(i) <= 255:
                text = '\033[38;5;{}m{}'.format(int(i), text)
        if self._open:
            text = '\033[0m' + text
        self._open = len(decorS or backcS or textcS or backcN or textcN) > 0
        return text

    def __mod__(self, fmt):
        return self._fmt(fmt)

    def __add__(self, text):
        return self._fmt() + text

    def __radd__(self, text):
        return text + self._fmt()

    def __str__(self):
        return self._fmt()

    def __call__(self, *args):
        if len(args) == 2:
            return self._fmt(args[1]) + args[0] + self._fmt()
        else:
            return self._fmt(args[0])

    def __repr__(self):
        return self._fmt()

color = color_class()


def getConfig(file):
    """ Open a yml parameter file
    """
    with open(file, 'r') as f:
        return yaml.load(f, loader)


def getParams(parameterfile, templatefile, required=None):
    """ Load parameters from a parameterfile and parameters missing from that from the templatefile. Raise an error when
        parameters in required are missing. Return a dictionary with the parameters.
    """
    params = getConfig(parameterfile)

    # recursively load more parameters from another file
    def moreParams(params, file):
        if not params.get('moreParams') == none():
            if os.path.isabs(params['moreParams']):
                moreParamsFile = params['moreParams']
            else:
                moreParamsFile = os.path.join(os.path.dirname(os.path.abspath(file)), params['moreParams'])
            print(color('Loading more parameters from {}'.format(moreParamsFile), 'g'))
            mparams = getConfig(moreParamsFile)
            moreParams(mparams, file)
            for k, v in mparams.items():
                if k not in params:
                    params[k] = v

    moreParams(params, parameterfile)

    #  convert string nones to type None
    for k, v in params.items():
        if v == none():
            params[k] = None

    if required is not None:
        for p in required:
            if p not in params:
                raise Exception('Parameter {} not given in parameter file.'.format(p))

    template = getConfig(templatefile)
    for k, v in template.items():
        if k not in params and not v == none():
            print(color('Parameter {} missing in parameter file, adding with default value: {}.'.format(k, v), 'r'))
            params[k] = v

    return params


def convertParamFile2YML(file):
    """ Convert a py parameter file into a yml file
    """
    with open(file, 'r') as f:
        lines = f.read(-1)
    with open(re.sub('\.py$', '.yml', file), 'w') as f:
        for line in lines.splitlines():
            if not re.match('^import', line):
                line = re.sub('(?<!#)\s*=\s*', ': ', line)
                line = re.sub('(?<!#);', '', line)
                f.write(line+'\n')


class objFromDict(dict):
    """ Usage: objFromDict(**dictionary).
        Print gives the list of attributes.
    """
    def __init__(self, **entries):
        super(objFromDict, self).__init__()
        for key, value in entries.items():
            key = key.replace('-', '_').replace('*', '_').replace('+', '_').replace('/', '_')
            self[key] = value

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        if key not in self:
            raise AttributeError(key)
        return self[key]

    def __repr__(self):
        return '** {} attr. --> '.format(self.__class__.__name__)+', '.join(filter((lambda s: (s[:2]+s[-2:]) != '____'),
                                                                                   self.keys()))

    def copy(self):
        return self.__deepcopy__()

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        copy = cls.__new__(cls)
        copy.update(**deepcopy(super(objFromDict, self), memodict))
        return copy

    def __dir__(self):
        return self.keys()


class none():
    """ special class to check if an object is some variation of none
    """
    def __eq__(self, other):
        if isinstance(other, none):
            return True
        if other is None:
            return True
        if hasattr(other, 'lower') and other.lower() == 'none':
            return True
        return False