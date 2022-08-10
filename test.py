import sys

def num_or_zero(s:str, default):
    if s.isdigit():
        return int(s)
    else:
        return 0

def get_args(argv):

    fname = None

    for i, arg in enumerate(argv):
        if arg == '-f' and i < len(argv)-1:
            fname = argv[i+1]
        elif arg == '-s' and i < len(argv)-1:
            step = num_or_zero(argv[i+1], step)
        elif arg == '-sv' and i < len(argv)-1:
            val_step = num_or_zero(argv[i+1], val_step)
        elif arg == '-imsize' and i < len(argv)-1:
            im_size = num_or_zero(argv[i+1], im_size)
        elif arg == '-e' and i < len(argv)-1:
            allowed_error = num_or_zero(argv[i+1], allowed_error)
        elif arg == '-ml' and i < len(argv)-1:
            min_length = num_or_zero(argv[i+1], min_length)

    if fname is None:
        print('please include \'-f [FILENAME]\'')
        exit()

    return fname, step, val_step, im_size, allowed_error, min_length

print(get_args(sys.argv))