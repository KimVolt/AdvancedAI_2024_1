import os
import argparse

from examples import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ex', type=int, default=0)
    args = parser.parse_args()

    if args.ex == 0:
        dice()
    elif args.ex == 1:
        print("sample: ", sample_dice(dices=2))
    elif args.ex == 2:
        print("ev_sample: ", ev_sample_dice(dices=2, trial=1000))
    elif args.ex == 3:
        print("increment_ev_sample: ", increment_ev_sample_dice(dices=2, trial=1000))
    elif args.ex == 4:
        example_step()
    elif args.ex == 5:
        agent_train()
    elif args.ex == 6:
        mcagent_train()
    elif args.ex == 7:
        comparison_normal_smapling()
    elif args.ex == 8:
        importance_sampling()
    elif args.ex == 9:
        revised_importance_sampling()
    else:
        print('Unknown example')
        exit(1)
        

    
    
