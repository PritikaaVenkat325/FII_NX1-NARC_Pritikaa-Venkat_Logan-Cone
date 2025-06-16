import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

# Pass the name of the scheduler (ie. mono_lin)
parser.add_argument('--schedule', default='', type=str)

def graph_beta_values(args):
    with open('logs/ptb/vae_' + args.schedule + '/log.txt', 'r') as file:
        batch = []
        beta = []
        i = 0
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

                # Get the beta value of the line
                curBeta = line[line.find("Beta: ") + 6 : line.find("Beta: ") + 12]
                curBeta = float(curBeta)
                

                batch.append(totalBatch)
                beta.append(curBeta)

        plt.plot(batch, beta)
        plt.xlabel("Batches")
        plt.ylabel("Beta")
        plt.show(block=True)
                


if __name__ == "__main__":
    args = parser.parse_args()
    graph_beta_values(args)


