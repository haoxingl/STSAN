class EarlystopHelper:
    def __init__(self, patiences=[5, 10], thres=0.1, error_delta=0, in_weight=0.5, out_weight=0.5):
        assert len(patiences) == 2
        self.patiences = patiences
        self.thres = thres
        self.error_delta = error_delta
        self.epoch_count = 0
        self.best_rmse = 2000.0
        self.best_epoch = None
        self.last_rmse = None
        self.thres_cnt = 0
        self.check_flag = False
        self.in_weight = in_weight
        self.out_weight = out_weight

    def refresh_status(self, eval_rmse):
        if not self.check_flag:
            if not self.last_rmse:
                self.last_rmse = eval_rmse
                return False
            else:
                if (self.last_rmse - eval_rmse)/self.last_rmse <= self.thres:
                    self.thres_cnt += 1
                    self.last_rmse = eval_rmse
                    if self.thres_cnt >= self.patiences[0]:
                        self.check_flag = True
                        return True
                    else:
                        return False
                else:
                    self.thres_cnt = 0
                    self.last_rmse = eval_rmse
                    return False
        else:
            return True

    def check(self, test_rmse, epoch):

        if self.check_flag:
            if test_rmse >= self.best_rmse * (1 + self.error_delta):
                self.epoch_count += 1
            else:
                self.epoch_count = 0
                self.best_rmse = test_rmse
                self.best_epoch = epoch + 1

            if self.epoch_count >= self.patiences[1]:
                return True
            else:
                return False

        else:
            return False

    def get_bestepoch(self):
        return self.best_epoch