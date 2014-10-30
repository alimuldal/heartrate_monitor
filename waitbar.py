from sys import stdout
from time import time

class Waitbar(object):
    def __init__(   self, amount=0., barwidth=50, totalwidth=75, title=None,
            showETA=False):

        self.showETA = showETA
        self._bw = barwidth
        self._tw = totalwidth
        self._stime = time()
        self._bar = ''
        self._eta = ''

        if title is not None:
            print title

        self.update(amount)

    def update(self,newamount):

        self._done = newamount

        n = int(round(self._done * self._bw))
        # bar = u'\u25AE'*n
        bar = '#' * n
        pad = '-' * (self._bw - n)
        self._bar = '[' + bar + pad + '] %2i%%' % (self._done * 100.)

        if self.showETA:
            if self._done == 0:
                self._eta = '  ETA: ?'
            else:
                dt = time() - self._stime
                eta = (dt / self._done) * (1. - self._done)
                self._eta = '  ETA: %s' %s2h(eta)
        self.display()

    def display(self):
        stdout.write('\r' + ' '*self._tw)
        if self._done == 1.:
            ftime = s2h(time() - self._stime)
            stdout.write('\r>>> Completed: %s\n' %ftime)
        else:
            nspace = max(
                (0,self._tw - (len(self._bar) + len(self._eta)))
                )
            stdout.write('\r' + self._bar + self._eta + ' '*nspace)

        stdout.flush()

class ElapsedTimer(object):

    def __init__(self, title=None, width=75):
        self.title = title
        self._npad = width - len(title)

    def start(self):
        self._stime = time()
        stdout.write('\r' + self.title + ' ' * self._npad)
        stdout.flush()

    def done(self):
        elapsed = s2h(time() - self._stime)
        donestr = 'done: ' + elapsed + '\n'
        stdout.write('\r' + self.title + ' '* self._npad + donestr)
        stdout.flush()

    def __enter__(self):
        self.start()

    def __exit__(self, *args):
        self.done()


def s2h(ss):
    mm,ss = divmod(ss,60)
    hh,mm = divmod(mm,60)
    dd,hh = divmod(hh,24)
    tstr = "%02i:%05.2f" %(mm,ss)
    if hh > 0:
        tstr = ("%02i:" %hh) + tstr
    if dd > 0:
        tstr = ("%id " %dd) + tstr
    return tstr