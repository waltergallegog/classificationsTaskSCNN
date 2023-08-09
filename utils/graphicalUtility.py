def progressBar(done, total, message=''):
    progress = 100*(done/total)
    bar = '#'*int(progress)+' '*(100-int(progress))
    print(f'\r{message}\t|{bar}| {progress:.02f}%', end='\r')
    if progress == 100:
        print('')
