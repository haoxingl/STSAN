class ReshuffleHelper:
    def __init__(self, es_patience=15, thres=[0.8, 1.3, 1.7]):
        self.es_patience = es_patience
        self.thres = thres
        self.flags = [False for _ in range(len(thres))]

    def check(self, epoch):
        epoch += 1
        for (index, flag) in enumerate(self.flags):
            if not flag:
                if epoch == int(self.es_patience * self.thres[index]):
                    self.flags[index] = True
                    if index < len(self.thres) - 1:
                        self.thres[index + 1] += self.thres[index]
                    print("Reshuffling...\n")
                    return True
                else:
                    return False

        return False
