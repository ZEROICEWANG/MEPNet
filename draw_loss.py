import matplotlib.pyplot as plt
import os

if __name__=='__main__':
    path='logs/2024_01_14_16_35_50/train_RES_Model2024_01_14_16_35_50.txt'
    target_loss=['L4-D','L4-S','L4-F','Los']
    # target_loss=['L3-S','L3-F']
    # target_loss=['L1-C']
    with open(path,'r') as f:
        lines=f.readlines()
    last_one=''
    loss_dict={target:[] for target in target_loss}
    for line in lines:
        if 'Epoch' in line:
            fields=last_one.split(',')[1:]
            for field in fields:
                for target in target_loss:
                    if target in field:
                        loss=field.split(":")[-1].strip()
                        loss_dict[target].append(float(loss))
        else:
            last_one=line
    
    for i,target in enumerate(target_loss):
        plt.figure(i)
        plt.plot([i for i in range(len(loss_dict[target]))],loss_dict[target])
        plt.legend([target+':%.4f'%min(loss_dict[target])])
        plt.savefig(os.path.join(*(path.split('/')[:-1]),'%s.png'%target))
        plt.close()
    # plt.show()