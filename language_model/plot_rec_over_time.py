import matplotlib.pyplot as plt


# List of directories to look through
dirs = ['logs/ptb/vae_cyc_lin/log.txt', 'logs/ptb/vae_cyc_sig/log.txt', 'logs/ptb/vae_cyc_quad/log.txt']


def graph_rec_values():
    for dir in dirs:
        with open(dir, 'r') as file:
            batch = []
            rec = []
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
                    curREC = line[line.find("TrainVAE_REC: ") + 14 : line.find("TrainVAE_REC: ") + 20]
                    curREC = float(curREC)
                    curREC

                    batch.append(totalBatch)
                    rec.append(curREC)
            plt.plot(batch, rec, label=dir[13:20])

    plt.xlabel("Batches")
    plt.ylabel("Rec")
    plt.title("Reconstruction Loss")
    plt.legend()
    plt.show(block=True)
                


if __name__ == "__main__":
    graph_rec_values()


