class ReshuffleHelper:
    def __init__(self, es_patience=15, thresholds=[0.8, 1.3, 1.7]):
        self.es_patience = es_patience
        self.thresholds = thresholds
        self.flags = [False for _ in range(len(thresholds))]

    def check(self, epoch):
        epoch += 1
        for (index, flag) in enumerate(self.flags):
            if not flag:
                if epoch == int(self.es_patience * self.thresholds[index]):
                    self.flags[index] = True
                    if index < len(self.thresholds) - 1:
                        self.thresholds[index + 1] += self.thresholds[index]
                    print("Reshuffling...\n")
                    return True
                else:
                    return False

        return False
