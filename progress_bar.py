import time

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


REPORT_ON = 0.3

class Progress:
    def __call__(self, iterable, description, total=None):
        if tqdm is None:
            try:
                total = len(iterable)
            except TypeError:
                total = -1

            start_time = time.time()
            last_report = start_time - REPORT_ON
            for n, el in enumerate(iterable):
                now = time.time()
                if now - last_report > REPORT_ON:
                    print("\r{}: {}/{}".format(description, n, total), end="")
                    last_report = now
                yield el
            print(" -> {}".format(time.time() - start_time))
        else:
            yield from tqdm(iterable, desc=description, total=total)


progress = Progress()

