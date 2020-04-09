import sys
import numpy as np

from preprocess import load_data


def fit(sequence, order=1, num_states=4):
    num_prev_states = num_states ** order
    counts = np.zeros([num_prev_states, num_states], dtype=int)
    
    for ind,seq in enumerate(sequence[order:]):
        prev_state = 0
        for i in range(order):
            prev_state += sequence[ind+i] * (num_states ** (order-i-1))
        counts[prev_state][seq] += 1

    transprob = np.zeros([num_prev_states, num_states])

    for i in range(num_prev_states):
        num_total = np.sum(counts[i])
        for j in range(num_states):
            if num_total != 0:
                transprob[i][j] = counts[i][j] / num_total

    return transprob


def calc_accu(seq_fit, seq_pred, num_days):
    transprob1 = fit(seq_fit[:,1].astype(int), order=1)
    transprob2 = fit(seq_fit[:,1].astype(int), order=2)
    transprob3 = fit(seq_fit[:,1].astype(int), order=3)

    transprob = np.zeros([4*4*4,4])
    for i in range(4*4*4):
        for j in range(4):
            prev1 = i % 4
            prev2 = i % 16
            transprob[i][j] = transprob3[i][j]*4 + transprob2[prev2][j]*2 + transprob1[prev1][j]
        transprob[i] /= np.sum(transprob[i])

    seq_fit = seq_fit[np.argsort(seq_fit[:,0])]

    num_total = 0
    num_correct = 0
    num_correct2 = 0

    for ind,seq in enumerate(seq_pred[3:]):
        time,act = seq
        prev1 = int(seq_pred[ind+2][1])
        prev2 = int(seq_pred[ind+1][1])*4 + prev1
        prev3 = int(seq_pred[ind][1])*16 + prev2

        """
        index = np.searchsorted(seq_fit[:,0], time)
        
        probs = transprob[prev3]
        for i in range(4):
            cnt = np.sum((np.abs(seq_fit[:,0][max(0,index-num_days):min(index+num_days,len(seq_fit))] - time) < 15)
                    & (seq_fit[:,1][max(0,index-num_days):min(index+num_days,len(seq_fit))] == i))
            probs[i] *= (2 ** (cnt / num_days))
        """
        num_total += 1
        pred = np.argmax(transprob[prev3])
        num_correct += (pred==act)
        
        """
        cur_state = prev1
        num_intervals = 1
        prob = transprob[prev3][cur_state]
        while (prob > 0.5):
            num_intervals += 1
            prev3 = prev2*4 + prev1
            prev2 = prev1*4 + prev1
            probs = transprob[prev3]
            print(probs)
            time += 240
            index = np.searchsorted(seq_fit[:,0],time)
            for i in range(4):
                cnt = np.sum((np.abs(seq_fit[:,0][max(0,index-num_days):min(index+num_days,len(seq_fit))] - time) < 15)
                        & (seq_fit[:,1][max(0,index-num_days):min(index+num_days,len(seq_fit))] == i))
                probs[i] *= (2 ** (cnt / num_days))
            probs /= np.sum(probs)
            prob *= probs[cur_state]#transprob[prev3][cur_state]
        if ind+2+num_intervals < len(seq_pred) and np.sum(seq_pred[ind+3:ind+2+num_intervals,1]!=cur_state) == 0 and seq_pred[ind+2+num_intervals][1] != cur_state:
            num_correct2 += 1
        """    

    return num_correct / num_total


def cross_validate(user, split_ratio=0.5):
    sequence, num_days = user
    sequence = np.asarray(sequence)
    split = int(len(sequence)*split_ratio)

    accu1 = calc_accu(sequence[0:split], sequence[split:], num_days//2)
    accu2 = calc_accu(sequence[split:], sequence[0:split], num_days//2)

    return (accu1 + accu2) / 2.


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        users = load_data(folder=path)
    else:
        users = load_data()
    accuracy = []
    for user in users:
        accuracy.append(cross_validate(user))

    print("mean accuracy:", np.mean(accuracy))

if __name__ == '__main__':
    main()
