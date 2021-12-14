import sys
IS_WINDOWS = sys.platform.startswith('win')

if IS_WINDOWS:
    import dill
    def run_dill(payload, args):
        fun = dill.loads(payload)
        return fun(*args)

    def make_dill(f,a):
        return (dill.dumps(f),(a,))