import multiprocessing as mp
import time
import sys

def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix),end='', flush=True)
    # Print New Line on Complete
    if iteration == total:
        print()


def create_multiproc_bar(total):
    q = mp.Queue()
    p = mp.Process(target=calib_printer_worker, args=(q, total))
    p.start()
    inc_fn = lambda x: q.put(x)
    kill_fn = p.terminate
    return inc_fn, kill_fn

def calib_printer_worker(q, total):
    iteration=0
    n_found=0
    while iteration<total:
        time.sleep(0.5)
        n_found += q.get()
        iteration+=1
        print_progress_bar(iteration, total, prefix='Progress:', suffix=f"Complete (calibration board found in {n_found}/{iteration} frames)", length=50)