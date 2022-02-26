import time

class run_time:
    def __init__(self, func = None):
        self.func = func

    def __call__(self, *args, **kwargs):
        start = time.time()
        res = self.func(*args, **kwargs)
        end = time.time()
        print(f"time: {end - start}")
        return res

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
    
    def get_time(self):
        return self.end - self.start
