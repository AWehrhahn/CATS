from os.path import join


class Step:
    def __init__(self, raw_dir, medium_dir, done_dir, config=None):
        self.raw_dir = raw_dir
        self.medium_dir = medium_dir
        self.done_dir = done_dir

        if config is not None:
            for key, value in config.items():
                setattr(self, key, value)

    def run(self, *args, **kwargs):
        raise NotImplementedError


class StepIO:
    filename = None

    @property
    def savefilename(self):
        return join(self.medium_dir, self.filename)

    def save(self, data, filename=None):
        raise NotImplementedError

    def load(self, filename=None):
        raise NotImplementedError
