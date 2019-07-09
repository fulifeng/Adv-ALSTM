from datetime import datetime
import numpy as np
import os


def load_cla_data(data_path, tra_date, val_date, tes_date, seq=2,
                  date_format='%Y-%m-%d'):
    fnames = [fname for fname in os.listdir(data_path) if
              os.path.isfile(os.path.join(data_path, fname))]
    print(len(fnames), ' tickers selected')

    data_EOD = []
    for index, fname in enumerate(fnames):
        # print(fname)
        single_EOD = np.genfromtxt(
            os.path.join(data_path, fname), dtype=float, delimiter=',',
            skip_header=False
        )
        # print('data shape:', single_EOD.shape)
        data_EOD.append(single_EOD)
    fea_dim = data_EOD[0].shape[1] - 2

    trading_dates = np.genfromtxt(
        os.path.join(data_path, '..', 'trading_dates.csv'), dtype=str,
        delimiter=',', skip_header=False
    )
    print(len(trading_dates), 'trading dates:')

    # transform the trading dates into a dictionary with index, at the same
    # time, transform the indices into a dictionary with weekdays
    dates_index = {}
    # indices_weekday = {}
    data_wd = np.zeros([len(trading_dates), 5], dtype=float)
    wd_encodings = np.identity(5, dtype=float)
    for index, date in enumerate(trading_dates):
        dates_index[date] = index
        # indices_weekday[index] = datetime.strptime(date, date_format).weekday()
        data_wd[index] = wd_encodings[datetime.strptime(date, date_format).weekday()]

    tra_ind = dates_index[tra_date]
    val_ind = dates_index[val_date]
    tes_ind = dates_index[tes_date]
    print(tra_ind, val_ind, tes_ind)

    # count training, validation, and testing instances
    tra_num = 0
    val_num = 0
    tes_num = 0
    # training
    for date_ind in range(tra_ind, val_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8:
                if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                    tra_num += 1
    print(tra_num, ' training instances')

    # validation
    for date_ind in range(val_ind, tes_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8:
                if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                    val_num += 1
    print(val_num, ' validation instances')

    # testing
    for date_ind in range(tes_ind, len(trading_dates)):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8:
                if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                    tes_num += 1
    print(tes_num, ' testing instances')

    # generate training, validation, and testing instances
    # training
    tra_pv = np.zeros([tra_num, seq, fea_dim], dtype=float)
    tra_wd = np.zeros([tra_num, seq, 5], dtype=float)
    tra_gt = np.zeros([tra_num, 1], dtype=float)
    ins_ind = 0
    for date_ind in range(tra_ind, val_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8 and \
                    data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                tra_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, : -2]
                tra_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
                tra_gt[ins_ind, 0] = (data_EOD[tic_ind][date_ind][-2] + 1) / 2
                ins_ind += 1

    # validation
    val_pv = np.zeros([val_num, seq, fea_dim], dtype=float)
    val_wd = np.zeros([val_num, seq, 5], dtype=float)
    val_gt = np.zeros([val_num, 1], dtype=float)
    ins_ind = 0
    for date_ind in range(val_ind, tes_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8 and \
                            data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                val_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, :-2]
                val_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
                val_gt[ins_ind, 0] = (data_EOD[tic_ind][date_ind][-2] + 1) / 2
                ins_ind += 1

    # testing
    tes_pv = np.zeros([tes_num, seq, fea_dim], dtype=float)
    tes_wd = np.zeros([tes_num, seq, 5], dtype=float)
    tes_gt = np.zeros([tes_num, 1], dtype=float)
    ins_ind = 0
    for date_ind in range(tes_ind, len(trading_dates)):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8 and \
                            data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                tes_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, :-2]
                # # for the momentum indicator
                # tes_pv[ins_ind, -1, -1] = data_EOD[tic_ind][date_ind - 1, -1] - data_EOD[tic_ind][date_ind - 11, -1]
                tes_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
                tes_gt[ins_ind, 0] = (data_EOD[tic_ind][date_ind][-2] + 1) / 2
                ins_ind += 1
    return tra_pv, tra_wd, tra_gt, val_pv, val_wd, val_gt, tes_pv, tes_wd, tes_gt

def load_reg_data(data_path, tra_date, val_date, tes_date, seq=2,
                  date_format='%Y-%m-%d', price_only=False):
    fnames = [fname for fname in os.listdir(data_path) if
              os.path.isfile(os.path.join(data_path, fname))]
    print(len(fnames), ' tickers selected')

    data_EOD = []
    for index, fname in enumerate(fnames):
        single_EOD = np.genfromtxt(
            os.path.join(data_path, fname), dtype=float, delimiter=',',
            skip_header=False
        )
        # print('data shape:', single_EOD.shape)
        data_EOD.append(single_EOD)
    fea_dim = data_EOD[0].shape[1] - 2
    if price_only:
        fea_dim = 1

    trading_dates = np.genfromtxt(
        os.path.join(data_path, '..', 'trading_dates.csv'), dtype=str,
        delimiter=',', skip_header=False
    )
    print(len(trading_dates), 'trading dates:')

    # transform the trading dates into a dictionary with index, at the same
    # time, transform the indices into a dictionary with weekdays
    dates_index = {}
    # indices_weekday = {}
    data_wd = np.zeros([len(trading_dates), 5], dtype=float)
    wd_encodings = np.identity(5, dtype=float)
    for index, date in enumerate(trading_dates):
        dates_index[date] = index
        # indices_weekday[index] = datetime.strptime(date, date_format).weekday()
        data_wd[index] = wd_encodings[datetime.strptime(date, date_format).weekday()]

    tra_ind = dates_index[tra_date]
    val_ind = dates_index[val_date]
    tes_ind = dates_index[tes_date]

    # count training, validation, and testing instances
    tra_num = 0
    val_num = 0
    tes_num = 0
    # training
    for date_ind in range(tra_ind, val_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                tra_num += 1
    print(tra_num, ' training instances')

    # validation
    for date_ind in range(val_ind, tes_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                val_num += 1
    print(val_num, ' validation instances')

    # testing
    for date_ind in range(tes_ind, len(trading_dates)):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                tes_num += 1
    print(tes_num, ' testing instances')

    # generate training, validation, and testing instances
    # training
    tra_pv = np.zeros([tra_num, seq, fea_dim], dtype=float)
    tra_wd = np.zeros([tra_num, seq, 5], dtype=float)
    tra_gt = np.zeros([tra_num, 1], dtype=float)
    tra_ref = np.zeros([tra_num, 2], dtype=float)
    ins_ind = 0
    for date_ind in range(tra_ind, val_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                if price_only:
                    tra_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, -3: -2]
                else:
                    tra_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, : -2]
                tra_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
                tra_gt[ins_ind, 0] = data_EOD[tic_ind][date_ind][-3]
                tra_ref[ins_ind, :] = data_EOD[tic_ind][date_ind][-2:]
                ins_ind += 1

    # validation
    val_pv = np.zeros([val_num, seq, fea_dim], dtype=float)
    val_wd = np.zeros([val_num, seq, 5], dtype=float)
    val_gt = np.zeros([val_num, 1], dtype=float)
    val_ref = np.zeros([val_num, 2], dtype=float)
    ins_ind = 0
    for date_ind in range(val_ind, tes_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                if price_only:
                    val_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, -3: -2]
                else:
                    val_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, :-2]
                val_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
                val_gt[ins_ind, 0] = data_EOD[tic_ind][date_ind][-3]
                val_ref[ins_ind, :] = data_EOD[tic_ind][date_ind][-2:]
                ins_ind += 1

    # testing
    tes_pv = np.zeros([tes_num, seq, fea_dim], dtype=float)
    tes_wd = np.zeros([tes_num, seq, 5], dtype=float)
    tes_gt = np.zeros([tes_num, 1], dtype=float)
    tes_ref = np.zeros([tes_num, 2], dtype=float)
    ins_ind = 0
    for date_ind in range(tes_ind, len(trading_dates)):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                if price_only:
                    tes_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, -3:-2]
                else:
                    tes_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, :-2]
                tes_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
                tes_gt[ins_ind, 0] = data_EOD[tic_ind][date_ind][-3]
                tes_ref[ins_ind, :] = data_EOD[tic_ind][date_ind][-2:]
                ins_ind += 1
    return tra_pv, tra_wd, tra_gt, tra_ref, \
           val_pv, val_wd, val_gt, val_ref, \
           tes_pv, tes_wd, tes_gt, tes_ref

def load_s2s_data(data_path, tra_date, val_date, tes_date, seq=2,
                  date_format='%Y-%m-%d', step=1, price_only=False):
    fnames = [fname for fname in os.listdir(data_path) if
              os.path.isfile(os.path.join(data_path, fname))]
    print(len(fnames), ' tickers selected')

    data_EOD = []
    for index, fname in enumerate(fnames):
        # print(index, fname)
        single_EOD = np.genfromtxt(
            os.path.join(data_path, fname), dtype=float, delimiter=',',
            skip_header=False
        )
        # print('data shape:', single_EOD.shape)
        data_EOD.append(single_EOD)
    fea_dim = data_EOD[0].shape[1] - 2
    if price_only:
        fea_dim = 1
    dat_sta_ind = 0
    for single_EOD in data_EOD:
        for ind, pvs in enumerate(single_EOD):
            if np.min(pvs) < -123320:
                if ind > dat_sta_ind:
                    dat_sta_ind = ind
            else:
                break
    dat_sta_ind += 1

    trading_dates = np.genfromtxt(
        os.path.join(data_path, '..', 'trading_dates.csv'), dtype=str,
        delimiter=',', skip_header=False
    )
    print(len(trading_dates), 'trading dates:')

    # transform the trading dates into a dictionary with index, at the same
    # time, transform the indices into a dictionary with weekdays
    dates_index = {}
    # indices_weekday = {}
    data_wd = np.zeros([len(trading_dates), 5], dtype=float)
    wd_encodings = np.identity(5, dtype=float)
    for index, date in enumerate(trading_dates):
        dates_index[date] = index
        # indices_weekday[index] = datetime.strptime(date, date_format).weekday()
        data_wd[index] = wd_encodings[datetime.strptime(date, date_format).weekday()]

    tra_ind = dates_index[tra_date]
    if tra_ind < dat_sta_ind:
        tra_ind = dat_sta_ind
    val_ind = dates_index[val_date]
    tes_ind = dates_index[tes_date]

    # generate training, validation, and testing instances
    # training
    tra_pv = np.zeros([len(data_EOD), val_ind - tra_ind - step, fea_dim],
                      dtype=float)
    tra_wd = np.zeros([len(data_EOD), val_ind - tra_ind - step, 5],
                      dtype=float)
    tra_gt = np.zeros([len(data_EOD), val_ind - tra_ind - step, 1],
                      dtype=float)
    tra_ref = np.zeros([len(data_EOD), val_ind - tra_ind - step, 2],
                       dtype=float)
    for tic_ind in range(len(fnames)):
        if price_only:
            tra_pv[tic_ind] = data_EOD[tic_ind][tra_ind: val_ind - step, -3: -2]
        else:
            tra_pv[tic_ind] = data_EOD[tic_ind][tra_ind: val_ind - step, : -2]
        tra_wd[tic_ind] = data_wd[tra_ind: val_ind - step, :]
        tra_gt[tic_ind, :, :] = data_EOD[tic_ind][tra_ind + step: val_ind, -3: -2]
        tra_ref[tic_ind, :, :] = data_EOD[tic_ind][tra_ind + step: val_ind, -2:]

    # validation
    val_pv = np.zeros([len(data_EOD), tes_ind - tra_ind - step, fea_dim], dtype=float)
    val_wd = np.zeros([len(data_EOD), tes_ind - tra_ind - step, 5], dtype=float)
    val_gt = np.zeros([len(data_EOD), tes_ind - tra_ind - step, 1], dtype=float)
    val_ref = np.zeros([len(data_EOD), tes_ind - val_ind, 2], dtype=float)
    for tic_ind in range(len(fnames)):
        # filter out instances without length enough history
        if price_only:
            val_pv[tic_ind] = data_EOD[tic_ind][tra_ind: tes_ind - step, -3: -2]
        else:
            val_pv[tic_ind] = data_EOD[tic_ind][tra_ind: tes_ind - step, :-2]
        val_wd[tic_ind] = data_wd[tra_ind: tes_ind - step, :]
        val_gt[tic_ind, :, :] = data_EOD[tic_ind][tra_ind + step: tes_ind, -3: -2]
        val_ref[tic_ind, :, :] = data_EOD[tic_ind][val_ind: tes_ind, -2:]

    # testing
    tes_pv = np.zeros([len(data_EOD), len(trading_dates) - tra_ind - step, fea_dim], dtype=float)
    tes_wd = np.zeros([len(data_EOD), len(trading_dates) - tra_ind - step, 5], dtype=float)
    tes_gt = np.zeros([len(data_EOD), len(trading_dates) - tra_ind - step, 1], dtype=float)
    tes_ref = np.zeros([len(data_EOD), len(trading_dates) - tes_ind, 2], dtype=float)
    for tic_ind in range(len(fnames)):
        if price_only:
            tes_pv[tic_ind] = data_EOD[tic_ind][tra_ind: len(trading_dates) - step, -3:-2]
        else:
            tes_pv[tic_ind] = data_EOD[tic_ind][tra_ind: len(trading_dates) - step, :-2]
        tes_wd[tic_ind] = data_wd[tra_ind: len(trading_dates) - step, :]
        tes_gt[tic_ind, :, :] = data_EOD[tic_ind][tra_ind + step: len(trading_dates), -3: -2]
        tes_ref[tic_ind, :, :] = data_EOD[tic_ind][tes_ind: len(trading_dates), -2:]
    return tra_pv, tra_wd, tra_gt, tra_ref, \
           val_pv, val_wd, val_gt, val_ref, \
           tes_pv, tes_wd, tes_gt, tes_ref

if __name__ == '__main__':
    # TEST
    _, _, tra_gt, _, _, val_gt, _, _, tes_gt = load_cla_data(
        '/home/ffl/nus/MM/fintech/tweet_stock/data/stocknet-dataset/price/ourpped',
        '2014-01-02', '2015-08-03', '2015-10-01'
    )
    print(np.sum(tra_gt))
    print(np.sum(val_gt))
    print(np.sum(tes_gt))
    print(np.sum(tes_gt) / 3720)
    # load_reg_data(
    #     '/home/ffl/nus/MM/fintech/tweet_stock/data/kdd17/ourpped',
    #     '2007-01-03', '2015-01-02', '2016-01-04'
    # )
