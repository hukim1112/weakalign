from options.options import ArgumentParser
args,arg_groups = ArgumentParser(mode='train_weak').parse()

print(arg_groups['weak_loss'])