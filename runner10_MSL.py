import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from util3 import KpiReader, KpiReaderTrain, CategoriesSampler
from evaluate_pot import *
# from logger import Logger
import time
import random
import copy
import datetime
import torch.multiprocessing
from model10.AnomalyTransformer import AnomalyTransformer
from scipy.io import savemat
from sklearn import metrics

torch.multiprocessing.set_sharing_strategy('file_system')


def get_data_str():
    dat = datetime.datetime.now()
    str_box = '-'
    str_box += str(dat.year)
    str_box += '-'
    str_box += str(dat.month)
    str_box += '-'
    str_box += str(dat.day)
    str_box += '--'
    str_box += str(dat.hour)
    str_box += '-'
    str_box += str(dat.minute)
    str_box += '-'
    str_box += str(dat.second)
    str_box += '-'
    str_box += str(dat.microsecond)
    str_box += '/'
    return str_box


class Trainer(object):
    def __init__(self, args, model, model_test, train, trainloader, test_loader, test_loader_box_finetune,
                 log_path='log_trainer', log_file='loss', epochs=20, batch_size=1024, learning_rate=0.001,
                 checkpoints='kpi_model.path', checkpoints_interval=1, device=torch.device('cuda:0')):
        self.args = args
        self.trainloader = trainloader
        self.test_loader = test_loader
        self.test_loader_box_finetune = test_loader_box_finetune
        self.train = train
        self.log_path = log_path
        self.log_file = log_file
        self.start_epoch = 0
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.model.to(device)
        self.model_test = model_test
        self.model_test.to(device)

        self.learning_rate = learning_rate
        self.checkpoints = checkpoints
        self.checkpoints_interval = checkpoints_interval
        # self.optimizer = optim.Adam([{'params': self.model.inference.parameters(), 'lr': self.learning_rate},
        #                              {'params': self.model.generation.parameters(), 'lr': self.learning_rate},
        #                              {'params': self.model.MetaNet.parameters(), 'lr': self.args.learning_rate}], )
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        total = sum([param.nelement() for param in self.model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))

        self.epoch_losses = []
        self.loss = {}
        # self.logger = Logger(self.log_path, self.log_file)

        # self.save_dir = './resoult/' + 'rnner7' + str(self.args.test_train_epoch).rjust(3, '0') + get_data_str()
        dirstr = get_data_str()
        self.save_dir = './resoult/' + 'rnner' + dirstr + 'saveall/'
        self.save_mod_dir = './resoult/' + 'rnner' + dirstr + 'mod/'
        print('save dir: ' + self.save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.save_mod_dir):
            os.makedirs(self.save_mod_dir)

    def save_checkpoint(self, epoch):
        torch.save({'epoch': epoch + 1,
                    'temperature': self.model.temperature,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'losses': self.epoch_losses},
                   self.checkpoints + '_epochs{}.pth'.format(epoch + 1))

    def load_checkpoint(self, start_ep):
        try:
            print("Loading Chechpoint from ' {} '".format(self.checkpoints + '_epochs{}.pth'.format(start_ep)))
            checkpoint = torch.load(self.checkpoints + '_epochs{}.pth'.format(start_ep))
            self.start_epoch = checkpoint['epoch']
            self.model.temperature = checkpoint['temperature']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch_losses = checkpoint['losses']
            print("Resuming Training From Epoch {}".format(self.start_epoch))
        except:
            print("No Checkpoint Exists At '{}', Starting Fresh Training".format(self.checkpoints))
            self.start_epoch = 0

    def loglikelihood_last_timestamp(self, x, recon_x_mu, recon_x_logsigma):
        llh = -0.5 * torch.sum(torch.pow(((x.float() - recon_x_mu.float()) / torch.exp(recon_x_logsigma.float())),
                                         2) + 2 * recon_x_logsigma.float() + np.log(np.pi * 2), dim=1)
        return llh

    def loglikelihood_last_timestamp_mse(self, x, recon_x_mu, recon_x_logsigma):
        recon_x_mu = torch.mean(recon_x_mu, dim=1)
        llh = torch.mean(F.mse_loss(x, recon_x_mu, reduction='none'), dim=1)
        return llh

    def forward_test(self, data):
        with torch.no_grad():
            out_test, _, otloss, _, KL_ma, KL_z, other = self.model_test(data)
            return out_test, otloss, other

    def train_model(self):
        timeLast = 0
        bestF1 = 0
        bestF1_unfinetune = 0
        bestF1_OT = 0
        bestAUC = 0

        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            losses = []
            loss1s = []
            kl_mas = []
            kl_zs = []
            ot_loss = []
            mse_loss = []
            loss_metas = []
            kld_pis = []
            print("Running Epoch : {} time: {} min".format(epoch + 1, (time.time() - timeLast) / 60))
            timeLast = time.time()
            # num_sample = 0
            for i, dataitem in enumerate(self.trainloader):
                _, _, data = dataitem
                data = data.squeeze().to(torch.float32).to(self.device)
                # num_sample += len(data)
                self.optimizer.zero_grad()
                out, _, otloss, loss_mse, KL_ma, KL_z, _ = self.model(data)
                warnup_v = (epoch / 20.0) * (epoch <= 20.0) + 1.0 * (epoch > 20.0)
                loss1 = self.model.loss_fn(data, _, _, out, _)
                # loss1 = self.model.loss_fn_mse(data, out)
                # loss = loss1 + warnup_v * (self.args.ma_k * KL_ma + self.args.z_k * KL_z)
                loss = loss1 + warnup_v * (
                        self.args.ma_k * KL_ma + self.args.z_k * KL_z) + self.args.ot_k * otloss.mean() + self.args.mse_k * loss_mse
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                loss1s.append(loss1.item())
                kl_mas.append(KL_ma.item())
                kl_zs.append(KL_z.item())
                ot_loss.append(otloss.mean().item())
                mse_loss.append(loss_mse.item())
                # kl_mas.append(0)
                # kl_zs.append(0)
                # loss_metas.append(loss_meta.item())

            meanloss = np.mean(losses)
            meanloss1 = np.mean(loss1s)
            meanklma = np.mean(kl_mas)
            meanklz = np.mean(kl_zs)
            meanotloss = np.mean(ot_loss)
            meanmseloss = np.mean(mse_loss)

            # meanloss_meta = np.mean(loss_metas)
            # meanpi = np.mean(kld_pis)
            self.epoch_losses.append(meanloss1)
            print(
                "Epoch {} : Average Loss: {:0>8f} Loglikelihood: {:0>8f} KL C: {:0>8f} KL z: {:0>8f} OT loss: {:0>8f} MSE loss: {:0>8f}".format(
                    epoch + 1, meanloss, meanloss1, meanklma, meanklz, meanotloss, meanmseloss))
            print("Epoch {} : ma std: {:0>8f}".format(epoch + 1,
                                                      torch.std(self.model.decoder.OTmodel.mus, dim=1).mean().item()))
            self.loss['Epoch'] = epoch + 1
            self.loss['Avg_loss'] = meanloss1
            # self.logger.log_trainer(epoch + 1, self.loss)
            # print(70840*((time.time()-timeLast)/num_sample))

            # test part：
            if epoch % 1 == 0:
                self.model.eval()
                res_each_mach_TP_befor_finetune = []
                res_each_mach_TN_befor_finetune = []
                res_each_mach_FP_befor_finetune = []
                res_each_mach_FN_befor_finetune = []

                all_pred = []
                all_lable = []

                OT_each_mach_TP = []
                OT_each_mach_TN = []
                OT_each_mach_FP = []
                OT_each_mach_FN = []

                save_likehood = []
                save_likehood_unfinetune = []
                save_labels_unfinetune = []
                save_hpoo_box = []
                save_hpoo_box_unfinetune = []
                save_TPs = []
                save_TPs_unfinetune = []

                otloss_box = []
                data_box_plot = []
                p_xj_box_plot = []
                threshold = []
                threshold_ot = []
                time_box = 0

                self.model.decoder.OTmodel.init_mus_self()

                for i_test_data_loder in range(len(self.test_loader)):
                    # for i_test_data_loder in range(2):
                    # print(i_test_data_loder)
                    # for i_test_data_loder in range(2):
                    #     i_test_data_loder = 10
                    self.model_test.load_state_dict(self.model.state_dict())
                    paras = []
                    for name, p in self.model_test.decoder.OTmodel.named_parameters():
                        if name == 'mus_self':
                            paras.append(p)

                    optimizer_for_one = optim.Adam(paras, self.args.learning_rate_finetune)
                    data_tran_box_temp = []
                    loss_res_box = []
                    labels_box = []
                    hpoo_box = []
                    otloss_box_i = []
                    data_box_plot_i = []
                    p_xj_box_i = []
                    # timeLast = time.time()

                    for i, dataitem in enumerate(self.test_loader[i_test_data_loder], 1):
                        timestamps, labels, data = dataitem
                        data = data.squeeze().to(torch.float32).to(self.device)
                        if i < self.args.test_train_step:
                            data_tran_box_temp.append(data[0:self.args.test_train_step2])
                        elif i == self.args.test_train_step:
                            # temp_time = time.time()
                            self.model_test.train()
                            for test_train_epoch_i in range(self.args.test_train_epoch):
                                for test_train_data_i in range(len(data_tran_box_temp)):
                                    optimizer_for_one.zero_grad()
                                    out, _, otloss, loss_mse, KL_ma, KL_z, _ = self.model_test(
                                        data_tran_box_temp[test_train_data_i])

                                    loss1 = self.model_test.loss_fn(data_tran_box_temp[test_train_data_i], _, _, out, _)
                                    # loss1 = self.model.loss_fn_mse(data, out)
                                    loss = loss1 + (
                                            self.args.ma_k * KL_ma + self.args.z_k * KL_z) + self.args.ot_k * otloss.mean() + self.args.mse_k * loss_mse
                                    loss.backward()
                                    optimizer_for_one.step()
                            # print(temp_time - time.time())
                            # temp_time = time.time()
                        else:
                            self.model_test.eval()

                            out_test, otloss, other = self.forward_test(data)
                            # last_timestamp = timestamps[:, -1, -1, -1]
                            otloss_box_i.append(-1 * otloss)
                            data_box_plot_i.append(np.array(data.cpu().numpy())[:, -1, :])
                            label_last_timestamp_tensor = labels[:, -1, -1, -1]

                            llh_last_timestamp = self.loglikelihood_last_timestamp(data[:, -1, :],
                                                                                   out_test[:, -1, :],
                                                                                   torch.from_numpy(np.array(1)))
                            # llh_last_timestamp = self.loglikelihood_last_timestamp_mse(data[:, -1, :],
                            #                                                            out_test[:, :, :],
                            #                                                            torch.from_numpy(np.array(1)))
                            loss_res_box.append(llh_last_timestamp)
                            labels_box.append(label_last_timestamp_tensor)
                            p_xj_box_i.append(other['p_xj'][:, -1, :])
                    # print("max i is %d" % i)
                    # print(time.time()-timeLast)
                    otloss_box_i = torch.cat(otloss_box_i).cpu().numpy().astype(float)

                    data_box_plot.append(np.vstack(data_box_plot_i[0:-1]))
                    # loss_res_box = self.normalization(torch.cat(loss_res_box).cpu().numpy().astype(float))
                    loss_res_box = torch.cat(loss_res_box).cpu().numpy().astype(float)
                    labels_box = torch.cat(labels_box).cpu().numpy()
                    p_xj_box_plot.append(torch.cat(p_xj_box_i).cpu().numpy().astype(float))
                    time_box += time.time() - timeLast
                    for pot_i in range(1):
                        lenth_loss_res_step = int(len(loss_res_box) / 1)
                        loss_res_box_i = loss_res_box[pot_i * lenth_loss_res_step:(pot_i + 1) * lenth_loss_res_step]
                        labels_box_i = labels_box[pot_i * lenth_loss_res_step:(pot_i + 1) * lenth_loss_res_step]
                        best_valid_metrics_unfinetune = {}
                        pot_result = pot_eval(loss_res_box_i, loss_res_box_i, labels_box_i, level=self.args.level)
                        best_valid_metrics_unfinetune.update(pot_result)

                        save_likehood_unfinetune.append(loss_res_box)
                        save_labels_unfinetune.append(labels_box)
                        save_TPs_unfinetune.append(best_valid_metrics_unfinetune)
                        save_hpoo_box_unfinetune.append(hpoo_box)
                        threshold.append(best_valid_metrics_unfinetune['pot-threshold'])
                        otloss_box.append(otloss_box_i)

                        res_each_mach_TP_befor_finetune.append(best_valid_metrics_unfinetune['pot-TP'])
                        res_each_mach_TN_befor_finetune.append(best_valid_metrics_unfinetune['pot-TN'])
                        res_each_mach_FP_befor_finetune.append(best_valid_metrics_unfinetune['pot-FP'])
                        res_each_mach_FN_befor_finetune.append(best_valid_metrics_unfinetune['pot-FN'])

                        all_pred.append(best_valid_metrics_unfinetune['pred'])
                        all_lable.append(labels_box_i)

                    # # OT loss
                    # for pot_i in range(1):
                    #     lenth_loss_res_step = int(len(otloss_box_i) / 1)
                    #     loss_res_box_i = otloss_box_i[pot_i * lenth_loss_res_step:(pot_i + 1) * lenth_loss_res_step]
                    #     labels_box_i = labels_box[pot_i * lenth_loss_res_step:(pot_i + 1) * lenth_loss_res_step]
                    #     ot_metrics = {}
                    #     pot_result = pot_eval(loss_res_box_i, loss_res_box_i, labels_box_i, level=self.args.level2)
                    #     ot_metrics.update(pot_result)
                    #
                    #     threshold_ot.append(ot_metrics['pot-threshold'])
                    #
                    #     OT_each_mach_TP.append(ot_metrics['pot-TP'])
                    #     OT_each_mach_TN.append(ot_metrics['pot-TN'])
                    #     OT_each_mach_FP.append(ot_metrics['pot-FP'])
                    #     OT_each_mach_FN.append(ot_metrics['pot-FN'])

                TP = np.sum(res_each_mach_TP_befor_finetune)
                TN = np.sum(res_each_mach_TN_befor_finetune)
                FP = np.sum(res_each_mach_FP_befor_finetune)
                FN = np.sum(res_each_mach_FN_befor_finetune)

                precision_unfine = TP / (TP + FP + 0.00001)
                recall_unfine = TP / (TP + FN + 0.00001)
                f1_unfine = 2 * precision_unfine * recall_unfine / (precision_unfine + recall_unfine + 0.00001)

                # OT resoult
                TP = np.sum(OT_each_mach_TP)
                TN = np.sum(OT_each_mach_TN)
                FP = np.sum(OT_each_mach_FP)
                FN = np.sum(OT_each_mach_FN)

                precision_OT = TP / (TP + FP + 0.00001)
                recall_OT = TP / (TP + FN + 0.00001)
                f1_OT = 2 * precision_OT * recall_OT / (precision_OT + recall_OT + 0.00001)

                # if bestF1_OT < f1_OT:
                #     bestF1_OT = f1_OT

                all_lable = np.concatenate(all_lable)
                all_pred = np.concatenate(all_pred)
                auc = metrics.roc_auc_score(all_lable, all_pred)

                if bestAUC < auc:
                    bestAUC = auc

                if bestF1_unfinetune < f1_unfine:
                    bestF1_unfinetune = f1_unfine

                    att_save_dic = {}
                    for save_att_i in range(len(otloss_box)):
                        str_att = 'OTlosss' + str(save_att_i)
                        att_save_dic[str_att] = otloss_box[save_att_i]

                    for save_att_i in range(len(save_likehood_unfinetune)):
                        str_att = 'likehood' + str(save_att_i)
                        att_save_dic[str_att] = save_likehood_unfinetune[save_att_i]

                    for save_att_i in range(len(save_labels_unfinetune)):
                        str_att = 'lable' + str(save_att_i)
                        att_save_dic[str_att] = save_labels_unfinetune[save_att_i]

                    for save_att_i in range(len(data_box_plot)):
                        str_att = 'data' + str(save_att_i)
                        att_save_dic[str_att] = data_box_plot[save_att_i]

                    for save_att_i in range(len(threshold)):
                        str_att = 'threshold' + str(save_att_i)
                        att_save_dic[str_att] = threshold[save_att_i]

                    for save_att_i in range(len(threshold_ot)):
                        str_att = 'threshold_ot' + str(save_att_i)
                        att_save_dic[str_att] = threshold_ot[save_att_i]

                    for save_att_i in range(len(p_xj_box_plot)):
                        str_att = 'p_xj' + str(save_att_i)
                        att_save_dic[str_att] = p_xj_box_plot[save_att_i]

                    savemat(self.save_dir + 'saveAllmat' + str(epoch + 1).rjust(3, '0') + ".mat", att_save_dic)

                    # savemat("../plot_res/r2_best.mat", {"likehood2": loss_res_box})
                    # # torch.save(self.model, self.save_mod_dir + str(epoch) + '.pt')
                    # np.save(self.save_dir + 'saveAll' + str(epoch + 1).rjust(3, '0') + '.npy', save_all)
                if bestF1_OT < f1_OT:
                    bestF1_OT = f1_OT

                if bestF1_unfinetune < f1_unfine:
                    bestF1_unfinetune = f1_unfine

                print(
                    "Epoch {} : precision: {:0>8f} recall: {:0>8f} f1: {:0>8f} ".format(
                        epoch + 1,
                        precision_unfine,
                        recall_unfine,
                        f1_unfine))

                # print(
                #     "Epoch {} : (OT) precision: {:0>8f} recall: {:0>8f} f1: {:0>8f} ".format(
                #         epoch + 1,
                #         precision_OT,
                #         recall_OT,
                #         f1_OT))
                print("Epoch {} : AUC: {:0>8f} ".format(epoch + 1, auc))

                print("Epoch {} : Best f1: {:0>8f} Best OT f1: {:0>8f} Best AUC: {:0>8f}\n".format(epoch + 1,
                                                                                                   bestF1_unfinetune,
                                                                                                   bestF1_OT, bestAUC))
        print("Training is complete!")

    def normalization(self, data):
        M_m = np.max(data) - np.min(data)
        return (data - np.min(data)) / M_m


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run():
    parser = argparse.ArgumentParser()
    # data set
    parser.add_argument('--data_set', type=str, default='MSL')  # 'SMD', 'SMAP' or 'MSL'
    # GPU option
    parser.add_argument('--gpu_id', type=int, default=0)
    # Dataset options
    parser.add_argument('--Train_dataset_path', type=str, default='../data_preprocess/data_processed2/')
    parser.add_argument('--Test_dataset_path', type=str, default='../data_preprocess/data_processed/')
    parser.add_argument('--batch_size', type=int, default=256)  # 2048
    parser.add_argument('--Test_batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--T', type=int, default=20)
    parser.add_argument('--win_size', type=int, default=1)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--n', type=int, default=38)

    # Model options
    parser.add_argument('--enc_dec', type=str, default='CNN')

    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_model_concept', type=int, default=256)  # 512 is better
    parser.add_argument('--d_vae', type=int, default=512)
    parser.add_argument('--d_c', type=int, default=256)
    parser.add_argument('--ma_k', type=float, default=0.01)
    parser.add_argument('--z_k', type=float, default=0.01)
    parser.add_argument('--ot_k', type=float, default=0.01)
    parser.add_argument('--mse_k', type=float, default=0)
    parser.add_argument('--layer_Transformer', type=int, default=3)

    parser.add_argument('--n_concepts', type=int, default=10)
    parser.add_argument('--n_privately_concepts', type=int, default=2)
    parser.add_argument('--softmax_t', type=float, default=1)

    # Training options
    parser.add_argument('--learning_rate', type=float, default=0.00002)
    parser.add_argument('--learning_rate_finetune', type=float, default=0.00002)
    parser.add_argument('--epochs', type=int, default=350)

    parser.add_argument('--level', type=float, default=0.01)  # 1.3
    parser.add_argument('--level2', type=float, default=0.01)

    parser.add_argument('--seed', type=float, default=42)

    # Meta options
    parser.add_argument('--test_train_step', type=int, default=2)
    parser.add_argument('--test_train_step2', type=int, default=10)
    parser.add_argument('--test_train_epoch', type=int, default=10)
    parser.add_argument('--meta_data_depart', type=float, default=0.8)  # train number
    parser.add_argument('--test_train_banchsize', type=int, default=256)
    parser.add_argument('--parallel_h_banch', type=int, default=32)
    parser.add_argument('--model_state', type=str, default='train')

    # OT options
    parser.add_argument('--lam', type=float, default=10)
    args = parser.parse_args()

    args.parallel_h_banch_size = int(args.test_train_banchsize / args.parallel_h_banch)
    args.seed = np.random.randint(2147483647)
    args.seed = 1044414390

    # 'SMD', 'SMAP' or 'MSL'
    if args.data_set == 'SMD':
        pass
    elif args.data_set == 'MSL':
        args.Train_dataset_path = './MSL_SMAP/train/'
        args.Test_dataset_path = './MSL_SMAP/test/'
        args.input_c = 55
        args.output_c = 55
        args.n = 55

        pass

    set_seed(args.seed)
    print(args)
    # Set up GPU
    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device('cuda:%d' % args.gpu_id)
    elif args.gpu_id < 0:
        device = torch.device('mps:0')
    else:
        device = torch.device('cpu')


    files = os.listdir(args.Train_dataset_path)
    if args.data_set == 'SMD':
        files = [args.Train_dataset_path + files[i] for i in range(len(files)) if files[i].split('-')[0] == 'machine']
    else:
        files = [args.Train_dataset_path + files[i] for i in range(len(files))]
    files.sort()
    files = files[0:int(len(files) * args.meta_data_depart)]
    kpi_value_train = KpiReaderTrain(files)
    sampler = CategoriesSampler(
        label_list=kpi_value_train.label,
        label_num=len(files),
        episode_size=args.parallel_h_banch_size,
        episode_num=(int(701817 / (args.batch_size * args.T))),
        # episode_num=int(len(kpi_value_train) / (args.batch_size))*args.parallel_h_banch_size,
        # episode_num=int(len(kpi_value_train) / (args.batch_size)) * args.parallel_h_banch_size,
        way_num=1,
        image_num=args.parallel_h_banch,
    )
    dataloader = torch.utils.data.DataLoader(
        kpi_value_train,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    files_test = os.listdir(args.Test_dataset_path)
    if args.data_set == 'SMD':
        files_test = [args.Test_dataset_path + files_test[i] for i in range(len(files_test)) if
                      files_test[i].split('-')[0] == 'machine']
    else:
        files_test = [args.Test_dataset_path + files_test[i] for i in range(len(files_test))]
    files_test.sort()
    files_test = files_test[int(len(files_test) * args.meta_data_depart):]

    test_loader_box = []
    for i in range(len(files_test)):
        Test_dataset_path = files_test[i]

        kpi_value_test = KpiReader(Test_dataset_path)
        test_loader = torch.utils.data.DataLoader(kpi_value_test,
                                                  batch_size=args.Test_batch_size,
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=args.num_workers)
        test_loader_box.append(test_loader)


    test_loader_box_finetune = []


    sgmvrnn = AnomalyTransformer(args=args, win_size=args.T, enc_in=args.input_c, d_model=args.d_model,
                                 c_out=args.output_c, e_layers=args.layer_Transformer, device=device,
                                 model_state='train')
    sgmvrnn_test = AnomalyTransformer(args=args, win_size=args.T, enc_in=args.input_c, d_model=args.d_model,
                                      c_out=args.output_c, e_layers=args.layer_Transformer, device=device,
                                      model_state='test')

    trainer = Trainer(args, sgmvrnn, sgmvrnn_test, kpi_value_train, dataloader, test_loader_box,
                      test_loader_box_finetune,
                      batch_size=args.batch_size,
                      epochs=args.epochs,
                      learning_rate=args.learning_rate,
                      device=device)

    trainer.train_model()


if __name__ == '__main__':
    run()
