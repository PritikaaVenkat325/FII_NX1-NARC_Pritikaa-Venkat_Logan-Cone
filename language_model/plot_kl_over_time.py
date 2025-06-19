import matplotlib.pyplot as plt


# List of directories to look through
dirs = ['logs/ptb/vae_cyc_lin/log.txt', 'logs/ptb/vae_cyc_sig/log.txt', 'logs/ptb/vae_cyc_quad/log.txt']


def graph_kl_values():
    for dir in dirs:
        with open(dir, 'r') as file:
            batch = []
            kl = []
            for line in file:
                if "Iters" in line:
                    # Get the epoch of the line
                    epoch = line[line.find("Epoch: ") + 7 : line.find("Epoch: ") + 9]
                    if(',' in epoch):
                        epoch = epoch[0]
                    epoch = int(epoch)

                    # Get the batch of the line
                    curBatch = line[line.find("Batch: ") + 7 : line.find("Batch: ") + 11]
                    if('/' in curBatch):
                        curBatch = curBatch[0:3]
                    curBatch = int(curBatch)

                    # Calculate the total batch
                    totalBatch = (epoch - 1) * 1359 + curBatch

                    # Get the kl value of the line
                    curKL = line[line.find("TrainVAE_KL: ") + 13 : line.find("TrainVAE_KL: ") + 19]
                    curKL = float(curKL)
                    

                    batch.append(totalBatch)
                    kl.append(curKL)
            plt.plot(batch, kl, label=dir[13:20])

    plt.xlabel("Batches")
    plt.ylabel("KL")
    plt.title("KL")
    plt.legend()
    plt.show(block=True)
                


if __name__ == "__main__":
    graph_kl_values()


