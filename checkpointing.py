# -*- coding: utf-8 -*-

CHECKPOINT_FILENAME = '.checkpoint'


def read_checkpoint_run(tag=""):
    try:
        file = open(CHECKPOINT_FILENAME + tag)
        for line in file:
            run = line[0].strip()
            break
        return int(run)
    except IOError:
        return -1


def new_checkpoint(run, tag=""):
    checkpoint_file = open(CHECKPOINT_FILENAME + tag, 'w')
    checkpoint_file.write(str(run))
    checkpoint_file.close()
