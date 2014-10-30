import numpy as np
import matplotlib.pyplot as pp
from matplotlib import animation
import time
import subprocess as sp
from os import devnull
from waitbar import Waitbar, s2h


class Movie(object):

    def __init__(self, data, framerate=10, loop=True, timestamp=False,
                 capture_rate=1., useblit=True, **plotkwargs):

        assert data.ndim in (3, 4)

        self.data = data
        self.framerate = framerate
        self.loop = loop
        self.timestamp = timestamp
        self.timetext = None
        self.capture_rate = capture_rate
        self.plotkwargs = plotkwargs
        self.counter = 0
        self.animator = None
        self.useblit = useblit

        self.nframes = data.shape[0]
        self.msec = int((1. / framerate) * 1E3)
        self.hasplayed = np.zeros(self.nframes, np.bool)

        self.fig = pp.figure()
        self.axes = self.fig.add_axes((0.1, 0.1, 0.80, 0.80))

        self.drawlist = []

        self.draw_first_frame()
        self.fig.canvas.mpl_connect('resize_event', self.on_resize)
        self.reinit_animation()

        pass

    def reinit_animation(self):
        if self.animator:
            self.stop()
        self.counter = 0
        self.animator = animation.FuncAnimation(
            self.fig,
            self.next,
            save_count=self.data.shape[0],
            interval=self.msec,
            blit=self.useblit)
        pass

    def on_resize(self, event=None):
        # this was a waste of time - it doesn't get called when the
        # figure is resized whilst rapidly drawing frames. the problem
        # is that we need to make sure the event queue gets flushed
        # BEFORE drawing the next frame
        # print "resize"
        # self.stop()
        # self.fig.canvas.draw()
        # self.start()
        pass

    def next(self, dummy=None):

        self.draw_frame(self.counter)
        self.hasplayed[self.counter] = True

        # print self.counter
        self.counter += 1
        if self.counter >= self.data.shape[0]:
            self.counter = 0

        return self.drawlist

    def draw_frame(self, frameidx):
        # artists = []

        self.image.set_data(self.data[self.counter])
        # artists.append(self.image)

        if self.timestamp:
            text = s2h(frameidx / float(self.capture_rate))
            self.timetext.set_text(text)
            # artists.append(self.timetext)
        pass

    def draw_first_frame(self):
        clim = self.plotkwargs.pop('clim', None)
        if not clim:
            clim = (self.data.min(), self.data.max())
        self.image = self.axes.imshow(self.data[0], clim=clim,
                                      **self.plotkwargs)
        self.drawlist.append(self.image)
        if self.timestamp:
            xpos = self.data.shape[2] * 0.85
            ypos = self.data.shape[1] * 0.95
            text = s2h(self.counter / self.capture_rate)
            self.timetext = self.axes.text(
                xpos, ypos, text, color=[0.7, 0.7, 0.7], va='top', ha='left'
            )
            self.drawlist.append(self.timetext)
        self.fig.canvas.draw()
        pass

    def stop(self):
        self.animator.event_source.stop()
        pass

    def start(self):
        self.animator.event_source.start()
        pass

    def rewind(self):
        self.animator.event_source.stop()
        self.counter = -1
        self.fig.canvas.draw()
        pass

    def set_framerate(self, framerate):
        self.msec = int((1. / framerate) * 10 ** 3)
        self.animator.event_source.interval = self.msec
        self.reinit_animation()
        pass

    def set_data(self, data):
        self.data = data
        self.reinit_animation()
        pass

    def save(self, fname, codec='rawvideo', fps=None, width=800,
             makelog=False):

        if not np.all(self.hasplayed):
            raise Exception(
                "All frames need to have been displayed at "
                "least once before writing")
        if fps is None:
            fps = self.framerate

        # make sure we start with the first frame
        self.rewind()

        # need to disconnect the first draw callback, since we'll be
        # doing draws. otherwise, we'll end up starting the animation.
        if self.animator._first_draw_id is not None:
            self.animator._fig.canvas.mpl_disconnect(
                self.animator._first_draw_id)
            reconnect_first_draw = True
        else:
            reconnect_first_draw = False

        # input pixel dimensions
        w, h = self.fig.canvas.get_width_height()
        nframes = self.data.shape[0]

        # make sure that output width and height are divisible by 2
        # (some codecs don't like odd dimensions)
        figsize = self.fig.get_size_inches()
        aspect = figsize[1] / figsize[0]
        width = 2 * int(width / 2.)
        height = 2 * int((width / 2.) * aspect)

        # spawn an ffmpeg process - we're going to pipe the frames
        # directly to ffmpeg as .pngs
        cmd = ['ffmpeg', '-y',
               '-loglevel', 'verbose',

               '-f', 'rawvideo',
               '-pixel_format', 'rgb24',
               # '-f', 'image2pipe',
               # '-vcodec','png',
               '-r', '%d' % fps,
               '-s', '%ix%i' % (w, h),
               '-i', 'pipe:0',

               '-vcodec', codec,
               '-an',
               # '-b:v', '%ik' %kbitrate,
               '-s', '%ix%i' % (width, height),
               fname]

        # print ' '.join(cmd)

        if makelog:
            logfile = open(fname + '.log', 'w')
        else:
            logfile = open(devnull, 'w')

        proc = sp.Popen(cmd, shell=False, stdin=sp.PIPE,
                        stdout=logfile, stderr=logfile)

        # Render each frame, save it to the stdin of the spawned process
        wbh = Waitbar(title='Writing %s' % fname, showETA=True)
        for ii, data in enumerate(self.animator.new_saved_frame_seq()):
            self.animator._draw_next_frame(data, blit=True)
            # self.animator._fig.savefig(proc.stdin,format='png')
            proc.stdin.write(self.animator._fig.canvas.tostring_rgb())
            wbh.update((ii + 1.) / nframes)

        logfile.close()

        if reconnect_first_draw:
            drawid = self.animator._fig.canvas.mpl_connect(
                'draw_event', self.animator._start)
            self.animator._first_draw_id = drawid


def rescale_8bit(A):
    A = A.astype(np.float32)
    A -= A.min()
    A /= A.ptp()
    A *= 255
    return A.astype(np.uint8)


def array2avi(A, fname, fps=25, codec='rawvideo', codec_params='', width=800,
              makelog=False):

    if A.ndim == 3:
        fmt = 'gray'
    elif A.ndim == 4:
        if A.shape[3] == 3:
            fmt = 'rgb8'
        elif A.shape[3] == 4:
            fmt = 'argb'
        else:
            raise Exception('Invalid input 4th dimension - should'
                            ' be rgb or argb')
    else:
        raise Exception('Invalid number of input dimensions (should be'
                        ' [frame,row,col] or [frame,row,col,(a)rgb]')

    A = rescale_8bit(A)

    nframes, h, w = A.shape[:3]
    aspect = float(h) / w
    width = 2 * int(width / 2.)
    height = 2 * int((width / 2.) * aspect)

    cmd = ['ffmpeg', '-y',

           '-f', 'rawvideo',
           '-pix_fmt', fmt,
           '-r', '%d' % fps,
           '-s', '%ix%i' % (w, h),
           '-i', 'pipe:0',

           '-vcodec', codec,
           # '-b:v', '%ik' %kbitrate,
           '-s', '%ix%i' % (width, height),
           fname]

    if makelog:
        logfile = open(fname + '.log', 'w')
    else:
        logfile = open(devnull, 'w')

    proc = sp.Popen(cmd, shell=False, stdin=sp.PIPE, stdout=logfile,
                    stderr=logfile)

    wbh = Waitbar(title='Writing %s' % fname, showETA=True)
    for ii, frame in enumerate(A):
        proc.stdin.write(frame.tostring())
        wbh.update((ii + 1.) / nframes)

    proc.stdin.close()
    proc.terminate()
    logfile.close()
