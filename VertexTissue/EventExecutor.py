def EventExecutor(events, fired=None):
    if not fired:
        fired = [False for e in events]

    def listen_and_execute(t, *args):
        for i, evt in enumerate(events):
            if not fired[i] and t > evt[0]:
                fired[i] = True
                evt[1](*args)
                if len(evt) > 2:
                    print(evt[2])

    return listen_and_execute
