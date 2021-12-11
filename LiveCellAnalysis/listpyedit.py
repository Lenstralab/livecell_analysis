#!/usr/bin/python3.8

#in case of using parts in Python 2, listFile supports python2, but other parts might not
from __future__ import print_function, division

import os
import re
import sys
import tkinter as tk
import tkinter.filedialog
import yaml
from copy import deepcopy
from glob import glob
from inspect import stack
from numbers import Number
from tllab_common.wimread import imread
from tllab_common.misc import objFromDict

if hasattr(yaml, 'full_load'):
    yamlload = yaml.full_load
else:
    yamlload = yaml.load

class listFile(list):
    def __init__(self, file=None, items=None):
        self.file = None
        self.info = {}
        if not file is None:
            if isinstance(file, str) and os.path.exists(file) and not os.path.isdir(file):
                self.load(file)
                self.file = file
            elif isinstance(file, (list, tuple)) and len(file)==1:
                if isinstance(file[0], str) and os.path.exists(file[0]) and not os.path.isdir(file[0]):
                    self.load(file[0])
                    self.file = file[0]
                else:
                    self += file[0]
            else:
                self += file
        if not items is None:
            self.extend(items)

    def sort(self):
        key = [''.join([i[k] for k in ('rawPath', 'trk_r', 'trk_g')]) for i in self]
        idx = [i[0] for i in sorted(enumerate(key), key=lambda x:x[1])]
        tmp = self.copy()
        for i, j in enumerate(idx):
            self[i] = tmp[j]
        del tmp

    @property
    def on(self):
        """ Return a new listFile with only the entries not commented out with #'s. """
        return listFile([i for i in self if i.get('use', True)])

    @property
    def off(self):
        """ Return a new listFile with only the entries commented out with #'s. """
        return listFile([i for i in self if not i.get('use', True)])

    def load(self, file):
        del self[:]
        with open(file, 'r') as f:
            content = f.read()
        listFiles = re.findall('(?<!\[)\[.*\](?!\])', content, re.DOTALL) # everything between listFiles=[ ... ]
        if len(listFiles):
            listFiles = listFiles[0][1:-1]
            items = re.findall('#?[^{}]*{[^{}]*}', listFiles, re.DOTALL) # N x (comment + entry)
            for item in items:
                entry = re.findall('#?\s*{[^{}]*}', item, re.DOTALL)[0]
                comment = re.findall('^[^{}]*', item, re.DOTALL)[0]
                comment = re.sub('^,?[\r\n\s#]*', '', comment)
                comment = re.sub('[\r\n\s#]*$', '', comment)
                use = not entry.startswith('#')
                entry = re.sub('^(\s*#)+', '', entry, flags=re.MULTILINE)
                entry = re.sub('\t', ' ', entry, flags=re.MULTILINE)
                d = objFromDict(use=use, comment=comment)
                d.update(yamlload(entry))
                self.append(d)

    def getInfo(self, n):
        if not isinstance(n, str):
            n = self[n]['rawPath']
        if not n in self.info:
            with imread(n) as im:
                self.info[n] = im.__repr__()
        return self.info[n]

    def __repr__(self):
        """ This is also exactly as how the file will be saved. """
        s = '# coding: utf-8\nlistFiles=['
        for item in self:
            s += '\n    '
            if 'comment' in item:
                if len(item['comment']):
                    s += '# {}\n    '.format(item['comment'])
            if not item.get('use', True):
                s += '# '
            s += '{'
            for i, (k, v) in enumerate(item.items()):
                if not k in ('use', 'comment'):
                    s += "'{}': {},\n    ".format(k, v.__repr__())
                    if not item.get('use', True):
                        s += '# '
            s += '},\n'
        s += ']'
        return s

    def save(self, file=None):
        """ This just wraps __repr__ and saves its output to a file. """
        file = os.path.abspath(file or self.file)
        if file is None:
            raise ValueError('No filename found in either self.file or argument file.')
        self.file = file
        if not os.path.exists(os.path.split(file)[0]):
            os.makedirs(os.path.split(file)[0])
        with open(file, 'w') as f:
            print(self, file=f)

    def copy(self):
        return deepcopy(self)

    def __eq__(self, other):
        if len(self)==0 and len(other)==0:
            return True
        elif len(self) != len(other):
            return False
        else:
            for i, j in zip(self, other):
                for k in ('metadata', 'maxProj', 'rawPath', 'trk_g', 'trk_r', 'frameWindow', 'use'):
                    if i[k] != j[k]:
                        return False
                if 'comment' in i and 'comment' in j:
                    if i['comment'] != j['comment']:
                        return False
                elif 'comment' in i:
                    if i['comment'] != '':
                        return False
                elif 'comment' in j:
                    if j['comment'] != '':
                        return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return listFile(items=super(listFile, self).__getitem__(item))
        elif isinstance(item, (list, tuple)):
            return listFile(items=[super(listFile, self).__getitem__(i) for i in item])
        else:
            return super(listFile, self).__getitem__(item)

    def __add__(self, file):
        new = self.copy()
        new += file
        return new

    def __iadd__(self, file):
        """ Merges this instance with another onother one. """
        if isinstance(file, listFile): #add from another instance
            self.extend(file)
        elif isinstance(file, dict):
            self.append(file)
        elif isinstance(file, (list, tuple)):
            for f in file:
                self += f
        elif file.endswith('.list.py'): #add from a list.py file
            files = glob(file)
            for file in files:
                self += listFile(file)
        elif os.path.isdir(file): #add all .list.py's we can find in a folder and its subfolders
            files = glob(os.path.join(file, '**', '*.list.py'))
            for file in files:
                self += listFile(file)
        else: #add from a rawPath, trk_r or trk_g
            trk_g, trk_r = 2 * [None]
            if not file.endswith('.txt'):
                rawPath = file
            else:
                if file.endswith('_trk_results_green.txt'):
                    trk_g = file
                elif file.endswith('_trk_results_red.txt'):
                    trk_r = file
                rawPath = ffind(re.escape(os.path.split(re.findall('.*(?=_cellnr)', file)[0])[1]),
                         os.path.split(file)[0].replace('analysis', 'data'), 'once')

            d = {'use': True}
            with imread(rawPath) as im:
                if rawPath.endswith('.czi'):
                    d['metadata'] = rawPath
                else:
                    d['metadata'] = rawPath
                if im.shape[3]==1:
                    d['maxProj'] = rawPath
                elif os.path.exists(os.path.splitext(rawPath)[0]+'_max.tif'):
                    d['maxProj'] = os.path.splitext(rawPath)[0]+'_max.tif'
                else:
                    d['maxProj'] = ''

                d['rawPath'] = rawPath
                d['trk_g'] = trk_g or ffind(re.escape(os.path.splitext(os.path.split(rawPath)[1])[0]) + '_cellnr_\d+_trk_results_green\.txt$',
                         os.path.split(rawPath)[0].replace('data', 'analysis'), 'once')
                d['trk_r'] = trk_r or ffind(re.escape(os.path.splitext(os.path.split(rawPath)[1])[0]) + '_cellnr_\d+_trk_results_red\.txt$',
                         os.path.split(rawPath)[0].replace('data', 'analysis'), 'once')
                d['frameWindow'] = [0, im.shape[4]-1]
            self.append(d)
        return self


def clip(v, min, max, wrap=False):
    if wrap:
        if min-max==1:
            return min
        else:
            return (v - min) % (1 + max - min) + min
    elif min<=v<=max:
        return v
    elif v<min:
        return min
    else:
        return max


class valueBox(tk.Frame):
    """ Text box with an integer and <> buttons. """
    def __init__(self, callback=None, value=0, min=0, max=0, wrap=False, **kwargs):
        super().__init__(**kwargs)
        self._value = clip(value, min, max)
        self._min = min
        self._max = max
        self.callback = callback

        self.bPrev = tk.Button(text='<', command=lambda: self.btn(-1, wrap), master=self)
        self.bPrev.grid(row=0, column=0)
        self.n = tk.IntVar(value=self.value)
        self.n.trace_add('write', self.change)
        self.N = tk.Entry(width=10, master=self, textvariable=self.n)
        self.N.grid(row=0, column=1)
        self.bNext = tk.Button(text='>', command=lambda: self.btn(1, wrap), master=self)
        self.bNext.grid(row=0, column=2)

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, value):
        self._min = value
        if self.value<value:
            self.value = value

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, value):
        self._max = value
        if self.value>value:
            self.value=value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        cv = clip(value, self.min, self.max)
        if not self._value==cv:
            self._value = cv
            if not self.callback is None:
                self.callback(self.value)
        if not self.N.get()==str(self._value):
            self.N.delete(0, tk.END)
            self.N.insert(0, str(self._value))

    def change(self, *args):
        n = self.N.get()
        if len(n):
            try:
                n = int(n)
            except:
                return
            self.value = n

    def btn(self, c, wrap):
        self.value = clip(self.value+c, self.min, self.max, wrap)


class CEntry(tk.Entry):
    """ tk.Entry with undo/redo functionality using Ctrl-z/Ctrl-y. """
    def __init__(self, *args, **kwargs):
        tk.Entry.__init__(self, *args, **kwargs)
        self.changes = [""]
        self.steps = int()
        self.bind("<Control-z>", self.undo)
        self.bind("<Control-y>", self.redo)
        self.bind("<Key>", self.add_changes)

    def undo(self, event=None):
        if self.steps != 0:
            self.steps -= 1
            self.delete(0, tk.END)
            self.insert(tk.END, self.changes[self.steps])

    def redo(self, event=None):
        if self.steps < len(self.changes):
            self.delete(0, tk.END)
            self.insert(tk.END, self.changes[self.steps])
            self.steps += 1

    def add_changes(self, event=None):
        if self.get() != self.changes[-1]:
            self.changes.append(self.get())
            self.steps += 1


class Message():
    def __init__(self, title, text, questions, main=False):
        self.answer = ''
        if main:
            self.window = tk.Tk()
        else:
            self.window = tk.Toplevel()
        self.window.title(title)
        tk.Label(self.window, text=text).pack()
        f = tk.Frame(self.window, height=25)
        f.pack()
        for q in questions:
            tk.Button(f, text=q, command=self.bPress(q)).pack(side=tk.LEFT)
        if main:
            self.window.mainloop()

    def bPress(self, n):
        def fun(*args):
            self.answer = n
            self.window.destroy()
        return fun


class app():
    """ GUI to view and edit .list.py files.

    Functionality:
        Browse through all entries in a .list.py file;  <>  buttons.
        Add/merge .list.py files;                        +  button.
        Remove entry;                                    -  button.
        Sort entries;                                  sort button.
        Edit rawPath, metadata, maxProj, trk_r, trk_g and frameWindow for each entry.
        Disable/enable entries: the entry will be preceded by #'s in the .list.py file when disabled.
        Add a comment to an entry (eg. why the entry is disabled).
        Display some metadata of the image file in rawPath.

    Keyboard shortcuts:
        Ctrl-N:          new file.
        Ctrl-O:          open a .list.py file.
        Alt-O:           open and merge all .list.py files in a folder.
        Ctrl-S:          save the .list.py file.
        Alt-S:           save the .list.py file and specify file name.
        PageUp/PageDown: move upwards/downwards in the .list.py file.
        Home/End:        move to the first/last entry in the .list.py file.

    Opening from terminal:
        Opening from terminal is more versatile than opening from the GUI because it supports globbing.
        ./listpyedit -h:                                  shows this documentation
        ./listpyedit:                                     opens a new .list.py
        ./listpyedit abc.list.py:                         opens abc.list.py
        ./listpyedit abc.list.py def.list.py:             opens and merges abc.list.py and def.list.py
        ./listpyedit /DATA/lenstra_lab/user/*.list.py:    opens and merges all list.py files in /DATA/lenstra_lab/user/
        ./listpyedit /DATA/lenstra_lab/user/**/*.list.py: opens and merges all list.py files in /DATA/lenstra_lab/user/ and subfolders

    wp@tl20200811
    """

    def __init__(self, files=None):
        self.n = 0
        if not files is None:
            self.list = listFile(files)
            self.lastFolder = os.path.split(files[0])[0]
        else:
            self.list = listFile()
            self.lastFolder = '/DATA/lenstra_lab'

        self.lastSavedList = self.list.copy()

        self.window = tk.Tk()
        self.window.bind('<Control-N>', self.new)
        self.window.bind('<Control-o>', self.add)
        self.window.bind('<Alt-o>', self.openFromFolder)
        self.window.bind('<Control-s>', self.save)
        self.window.bind('<Alt-s>', self.saveas)
        self.window.bind('<Prior>', lambda *args: self.nChangeKey(-1))
        self.window.bind('<Next>', lambda *args: self.nChangeKey(1))
        self.window.bind('<Home>', lambda *args: self.nChangeKey(-len(self.list)))
        self.window.bind('<End>', lambda *args: self.nChangeKey(len(self.list)))

        # create a pulldown menu, and add it to the menu bar
        menubar = tk.Menu(self.window)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="New", command=self.new, underline=0, accelerator='Ctrl+N')
        filemenu.add_command(label="Open", command=self.add, underline=0, accelerator='Ctrl+O')
        filemenu.add_command(label="Open all in folder", command=self.openFromFolder, accelerator='Alt+O')
        filemenu.add_command(label="Save", command=self.save, underline=0, accelerator='Ctrl+S')
        filemenu.add_command(label="Save as", command=self.saveas, accelerator='Alt+S')
        menubar.add_cascade(label="File", menu=filemenu)
        self.window.config(menu=menubar)

        f = tk.Frame(self.window, height=25)
        f.grid(row=0, column=0)

        # first row with # control and add/remove buttons
        self.UseVar = tk.BooleanVar()
        self.Use = tk.Checkbutton(f, text='use', command=self.useChange, variable=self.UseVar)
        self.Use.grid(row=0, column=0)
        self.total = tk.Entry(f, width=10)
        self.total.grid(row=0, column=1)
        self.N = valueBox(self.nChange, 1, 1, clip(len(self.list), 1, 1e6), True, height=25, master=f)
        self.N.grid(row=0, column=2)
        self.bAdd = tk.Button(f, text='+', command=self.add)
        self.bAdd.grid(row=0, column=3)
        self.bRm = tk.Button(f, text='-', command=self.rm)
        self.bRm.grid(row=0, column=4)
        self.bSort = tk.Button(f, text='sort', command=self.sort)
        self.bSort.grid(row=0, column=5)

        g = tk.Frame(self.window)
        g.grid(row=1, column=0)

        def field(k):
            t = tk.StringVar()
            t.trace('w', self.fieldChange(k, t))
            return (t, CEntry(g, width=150, textvariable=t))

        lbl = ('rawPath', 'metadata', 'maxProj', 'trk_r', 'trk_g', 'comment')
        self.lbl = {k: field(k) for k in lbl}

        for i, (k, v) in enumerate(self.lbl.items()):
            tk.Label(g, text=k).grid(row=i+1, column=0)
            v[1].grid(row=i+1, column=1)

        tk.Label(g, text='frameWindow').grid(row=7, column=0)
        h = tk.Frame(g, height=25)
        h.grid(row=7, column=1)
        self.fw = []
        for i in range(2):
            self.fw.append(valueBox(self.FWchange(i), i, 0, 1e6, height=25, master=h))
            self.fw[-1].grid(row=0, column=i)

        self.getInfo = tk.BooleanVar()
        getInfo = tk.Checkbutton(g, text='info', command=self.disp, variable=self.getInfo)
        getInfo.grid(row=8, column=0)
        self.info = tk.Label(g, width=150, justify=tk.LEFT, font=('Courier', 12))
        self.info.grid(row=8, column=1)

        self.disp()
        self.window.mainloop()
        self.askSave()

    @property
    def changed(self):
        if len(self.list)==0:
            return False
        elif self.list.file is None:
            return True
        else:
            return self.list != self.lastSavedList

    def askSave(self):
        if self.changed:
            save = Message('File was not saved.', 'Save the file?', ('Save', 'Save as', 'No'), main=not self.active).answer
            if save=='Save':
                self.save()
            elif save=='Save as':
                self.saveas()

    @property
    def active(self):
        try:
            self.window.winfo_exists()
            return True
        except:
            return False

    def new(self, *args):
        self.askSave()
        self.list = listFile()
        self.lastSavedList = self.list.copy()
        self.disp()

    def sort(self):
        self.list.sort()
        self.disp()

    def fieldChange(self, k, v):
        "One of the items in an entry is edited."
        def fun(*args):
            if len(self.list):
                if not k in self.list[self.n] or (k in self.list[self.n] and not self.list[self.n][k]==v.get()):
                    self.list[self.n][k] = v.get()
                self.disp()
        return fun

    def useChange(self):
        "Whether an entry is used (not commented out) or not is changed."
        if len(self.list):
            self.list[self.n]['use'] = self.UseVar.get()
            self.disp()

    def FWchange(self, n):
        "The frameWindow of an entry is changed."
        def fun(value):
            if len(self.list):
                if not 'frameWindow' in self.list[self.n]:
                    self.list[self.n]['frameWindow'] = [0, 0]
                self.list[self.n]['frameWindow'][n] = value
                self.disp()
        return fun

    def nChangeKey(self, value):
        "Which entry is displayed is changed by a keypress."
        self.N.value += value

    def nChange(self, value):
        "Which entry is displayed is changed."
        self.n = value-1
        self.disp()

    def rm(self):
        "An entry is removed."
        if len(self.list):
            del self.list[self.n]
            self.disp()

    def add(self, *args):
        "Extend the listfile by opening something."
        files = tk.filedialog.askopenfilenames(initialdir=self.lastFolder,
                                               filetypes=(('list.py', '*.list.py'), ('trk_r', '*_trk_results_red.txt'), ('trk_g', '*_trk_results_green.txt'), ('all files', '*')))
        if len(files):
            self.lastFolder = os.path.split(files[0])[0]
            for file in files:
                try:
                    self.list += file
                except Exception as e:
                    print(e)
            self.disp()

    def openFromFolder(self, *args):
        fld = tk.filedialog.askdirectory(initialdir=self.lastFolder)
        self.lastFolder = os.path.split(fld)[0]
        self.list += fld
        self.disp()

    def save(self, *args):
        if self.list.file is None:
            self.saveas()
        else:
            self.list.save()
        self.lastSavedList = self.list.copy()
        self.title()

    def saveas(self, *args):
        file = tk.filedialog.asksaveasfilename(defaultextension='.list.py', initialdir=self.lastFolder)
        self.list.save(file)
        self.lastSavedList = self.list.copy()
        self.title()

    def title(self):
        try:
            self.window.winfo_toplevel().title('listmod: {}{}'.format(self.changed * '*', self.list.file))
        except Exception as e:
            print(e)

    def disp(self):
        """ Display an entry, call this everytime something changes. """
        self.title()
        self.total.delete(0, tk.END)
        self.total.insert(0, '{} / {}'.format(len(self.list.on), len(self.list)))
        if not len(self.list):
            for v in self.lbl.values():
                v[0].set('')
            self.N.max = 1
        else:
            if not self.N.max==len(self.list):
                self.N.max = len(self.list)
            for k, v in self.lbl.items():
                if k in self.list[self.n]:
                    v[0].set(self.list[self.n][k])
                else:
                    v[0].set('')

            if 'frameWindow' in self.list[self.n]:
                fw = self.list[self.n]['frameWindow'].copy()
                for i in range(2):
                    self.fw[i].value = fw[i]
            if 'use' in self.list[self.n]:
                if self.list[self.n]['use']:
                    self.Use.select()
                else:
                    self.Use.deselect()
            if self.getInfo.get():
                self.info['text'] = re.sub('^#*\n', '', self.list.getInfo(self.n))
            else:
                self.info['text'] = ''


def ffind(expr, *args, **kwargs):
    """
    --------------------------------------------------------------------------
    % usage: fnames=ffind(expr,folder,rec,'once','directory')
    %
    % finds files that match regular expression 'expr' in 'folder' or
    % subdirectories
    %
    % inputs:
    %   folder: startfolder path, (optional, default: current working dirextory)
    %   expr:   string: regular expression to match
    %               example: to search for doc files do: '\.doc$'
    %           list or tuple: ffind will look for the folder in expr[0] starting
    %           from 'folder', 'rec' deep, then from that folder it will continue
    %           to navigate down to expr{end-1} and find the file (or folder) in
    %           expr[-1]
    %               example: ffind(('M_FvL','','^.*\.m'),'/home')
    %   rec:    recursive (optional, default: 3), also search in
    %           subdirectories rec deep (keep it low to avoid eternal loops)
    %   once:   optional flag, if enabled ffind will only output the first
    %           match it encounters as a string, use only if the existence of
    %           only one match is certain
    %   directory: optional flag: ffind only finds directories instead of files
    %
    % output:
    %   fnames: list containing all matches
    %
    % date: Aug 2014
    % author: wp
    % version: <01.00> (wp) <20140812.0000> Matlab implementation.
    %          <02.00>      <20180214.0000> Add once and directory flags, rec
    %                                       now signifies the folder-depth.
    %          <03.00>      <20190326.0000> Python implementation.
    %--------------------------------------------------------------------------
    """

    # argument parsing
    for i in args:
        if isinstance(i, Number):
            rec = i
        elif i == 'once':
            once = True
        elif i == 'directory':
            directory = True
        elif isinstance(i, str):
            folder = i

    for key, value in kwargs.items():
        if key == 'once':
            once = value
        elif key == 'directory':
            directory = value

    if not 'rec' in vars().keys():
        rec = 3
    if not 'once' in vars().keys():
        once = False
    if not 'directory' in vars().keys():
        directory = False
    if not 'folder' in vars().keys():
        folder = os.getcwd()

    if not folder[-1] == os.sep:
        folder = folder + os.sep

    if isinstance(expr, tuple):
        expr = list(expr)

    # search for the path in expr if needed
    if isinstance(expr, (list, tuple)):
        if len(expr) > 1:
            folder = ffind(expr[0], folder, rec, 'directory')
            for e in expr[1:-1]:
                folder_tmp = []
                for f in folder:
                    folder_tmp.extend(ffind(e, f, 0, 'directory'))
                folder = folder_tmp
            fnames = []
            for f in folder:
                fnames.extend(ffind(expr[-1], f, 0, once=once, directory=directory))
                if len(fnames) and once:
                    fnames = fnames[0]
                    break
            if not len(fnames) and once:
                fnames = ''
            return fnames
        expr = expr[0]

    # an empty expression should match everything
    if not isinstance(expr, re.Pattern):
        if not len(expr) > 0:
            expr = '.*'
        expr = re.compile(expr, re.IGNORECASE)

    try:
        lst = os.listdir(folder)
    except:
        lst = []

    fnames = []
    dirnames = []
    for l in lst:
        if re.search('^\.', l) != None:  # don't search in/for hidden things
            continue
        if (os.path.isdir(folder + l) == directory) & (re.search(expr, l) != None):
            fnames.append(folder + l)
            if once:
                break
        if rec and os.path.isdir(folder + l):  # list all folders, but don't go in them yet,
            dirnames.append(folder + l)        # faster when the target is in the current folder and once=True
    if not once or not len(fnames):  # recursively go through all subfolders
        for d in dirnames:
            fnames.extend(ffind(expr, d, rec - 1, once=once, directory=directory))
            if once and len(fnames):
                break

    if once and stack()[1][3] != 'ffind':
        if len(fnames):
            fnames = fnames[0]
        else:
            fnames = ''
    else:
        fnames.sort()
    return fnames


def main():
    if len(sys.argv)>1:
        if sys.argv[1] in ('-h', '--help'):
            print(app.__doc__)
            run = False
        else:
            files = sys.argv[1:]
            run = True
    else:
        files = None
        run = True
    if run:
        a = app(files)
    imread.kill_vm()

if __name__ == '__main__':
    main()