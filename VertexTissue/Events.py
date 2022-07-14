def EventExecutor(events, fired=None):
    if not fired:
        fired = [False]*len(events)

    def listen_and_execute(t, *args):
        something_fired = False
        for i, evt in enumerate(events):
            if not fired[i] and t >= evt[0]:
                fired[i] = True
                evt[1](*args)
                something_fired=True
                if len(evt) > 2:
                    print(evt[2])

        return something_fired

    return listen_and_execute


def EventListenerPair():

    fired=False

    def event(*_):
        nonlocal fired
        fired=True

    def listener(*_):
        nonlocal fired
        return fired

    return event, listener