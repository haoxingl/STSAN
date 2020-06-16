import json, codecs


class EarlystopHelper:
    def __init__(self, name, patiences=[5, 10], threshold=0.01, error_delta=0):
        assert len(patiences) == 2
        self.name = name
        self.patiences = patiences
        self.threshold = threshold
        self.error_delta = error_delta
        self.epoch_cnt = 0
        self.best_epoch = -1
        self.best_rmse = 1e6
        self.last_rmse = None
        self.start_cnt = 0
        self.check_flag = False
        self.stop_flag = False

    def refresh_status(self, eval_rmse):
        if self.check_flag:
            return True
        else:
            if not self.last_rmse:
                self.last_rmse = eval_rmse
                return False
            else:
                if (self.last_rmse - eval_rmse) / self.last_rmse <= self.threshold:
                    self.start_cnt += 1
                    self.last_rmse = eval_rmse
                    if self.start_cnt >= self.patiences[0]:
                        self.check_flag = True
                        return True
                    else:
                        return False
                else:
                    self.start_cnt = 0
                    self.last_rmse = eval_rmse
                    return False

    def check(self, test_rmse, epoch):

        if self.check_flag and not self.stop_flag:
            if test_rmse >= self.best_rmse * (1 + self.error_delta):
                self.epoch_cnt += 1
            else:
                self.epoch_cnt = 0
                self.best_rmse = test_rmse
                self.best_epoch = epoch + 1

            if self.epoch_cnt >= self.patiences[1]:
                self.stop_flag = True
                return self.stop_flag
            else:
                return False

        elif self.stop_flag:
            return True
        else:
            return False

    def get_bestepoch(self):
        return self.best_epoch

    def save_ckpt(self, path):
        ckpt_record = {
            'epoch_cnt': self.epoch_cnt,
            'best_epoch': self.best_epoch,
            'best_rmse': self.best_rmse,
            'last_rmse': self.last_rmse,
            'start_cnt': self.start_cnt,
            'check_flag': self.check_flag,
            'stop_flag': self.stop_flag
        }
        ckpt_record = json.dumps(ckpt_record, indent=4)
        with codecs.open(path + '/es_helper_{}.json'.format(self.name), 'w', 'utf-8') as outfile:
            outfile.write(ckpt_record)

    def load_ckpt(self, path):
        with codecs.open(path + '/es_helper_{}.json'.format(self.name), encoding='utf-8') as json_file:
            ckpt_record = json.load(json_file)
            self.epoch_cnt = ckpt_record['epoch_cnt']
            self.best_epoch = ckpt_record['best_epoch']
            self.best_rmse = ckpt_record['best_rmse']
            self.last_rmse = ckpt_record['last_rmse']
            self.start_cnt = ckpt_record['start_cnt']
            self.check_flag = ckpt_record['check_flag']
            self.stop_flag = ckpt_record['stop_flag']
