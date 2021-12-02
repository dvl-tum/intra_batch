import os.path as osp
import logging
import random
import net
import dataset
from RAdam import RAdam
from collections import defaultdict
import torch.nn as nn
from utils import losses
import torch
import random
import data_utility
import time
import torch.nn.functional as F
import copy
import sys
from evaluation import Evaluator_DML
import utils.utils as utils
import matplotlib.pyplot as plt
import os
import json
from torch import autograd
from tqdm import tqdm
autograd.set_detect_anomaly(True)

logger = logging.getLogger('GNNReID.Training')

torch.manual_seed(0)

class Trainer():
    def __init__(self, config, save_folder_nets, save_folder_results,
                 device, timer):
        self.config = config
        self.device = device
        self.save_folder_results = save_folder_results
        self.save_folder_nets = save_folder_nets + '_intermediate'
        utils.make_dir(save_folder_nets)
        self.save_folder_nets_final = save_folder_nets

        self.timer = timer
        self.fn = self.config['dataset'][
                      'dataset_short'] + '_intermediate_model_' + str(timer)
        self.net_type = self.config['models']['encoder_params']['net_type']
        self.dataset_short = self.config['dataset']['dataset_short']

        self.best_recall = 0
        self.best_hypers = None
        self.num_iter = 30 if 'hyper' in config['mode'].split('_') else 1
        print(torch.__version__)
        print(torch.version.cuda)
        import torch_scatter
        print(torch_scatter.__version__)
        print(torch.__file__)
        import sklearn
        print(sklearn.__version__)
        import torchvision
        print(torchvision.__version__)
        print(torchvision.__file__)
    def train(self):
        best_recall = 0

        for i in range(self.num_iter):
            logger.info("Search iter {}/{}\n{}\n{}".format(i+1, self.num_iter, self.config, self.timer))
            self.nb_clusters = self.config['train_params']['num_classes_iter']
            mode = 'neck_' if self.config['models']['encoder_params']['neck'] else ''
            
            self.update_params()
            
            self.encoder, sz_embed = net.load_net(
                    self.config['dataset']['dataset_short'],
                    self.config['dataset']['num_classes'],
                    self.config['mode'],
                    **self.config['models']['encoder_params'])
            
            self.encoder = self.encoder.to(self.device) 
            
            self.gnn = net.GNNReID(self.device, 
                    self.config['models']['gnn_params'], 
                    sz_embed).to(self.device)
            
            if self.config['models']['gnn_params']['pretrained_path'] != "no": 
                load_dict = torch.load(self.config['models']['gnn_params']['pretrained_path'], 
                        map_location='cpu')
                self.gnn.load_state_dict(load_dict)

            self.graph_generator = net.GraphGenerator(self.device, **self.config['graph_params'])
            self.evaluator = Evaluator_DML(nb_clusters=self.nb_clusters, 
                    dev=self.device, **self.config['eval_params'])
             
            params = list(set(self.encoder.parameters())) + list(set(self.gnn.parameters()))
            param_groups = [{'params': params,
                             'lr': self.config['train_params']['lr']}]

            self.opt = RAdam(param_groups,
                             weight_decay=self.config['train_params']['weight_decay'])

            self.get_loss_fn(self.config['train_params']['loss_fn'], 
                    self.config['dataset']['num_classes'])

            # Do training in mixed precision
            if self.config['train_params']['is_apex'] == 1:
                global amp
                from apex import amp
                [self.encoder, self.gnn], self.opt = amp.initialize([self.encoder, self.gnn], self.opt,
                                                        opt_level="O1")
            if torch.cuda.device_count() > 1:
                self.encoder = nn.DataParallel(self.encoder)

            self.get_data(self.config['dataset'], self.config['train_params'],
                          self.config['mode'])

            best_recall_iter, model = self.execute(
                self.config['train_params'],
                self.config['eval_params'])

            hypers = ', '.join(
                [k + ': ' + str(v) for k, v in self.config.items()])
            logger.info('Used Parameters: ' + hypers)

            logger.info('Best Recall: {}'.format(best_recall_iter))

            if best_recall_iter > best_recall and not 'test' in self.config['mode'].split('_'):
                os.rename(osp.join(self.save_folder_nets, self.fn + '.pth'),
                          osp.join(self.save_folder_nets_final, str(best_recall_iter) + mode + self.net_type + '_' +
                          self.dataset_short + '.pth'))
                os.rename(osp.join(self.save_folder_nets, 'gnn_' + self.fn + '.pth'),
                          osp.join(self.save_folder_nets_final, str(best_recall_iter) + 'gnn_' + mode + self.net_type + '_' +
                          self.dataset_short + '.pth'))
                best_recall = best_recall_iter
            elif 'test' in self.config['mode'].split('_'):
                best_recall = best_recall_iter
            
            best_hypers = ', '.join(
                    [str(k) + ': ' + str(v) for k, v in self.config.items()])

        logger.info("Best Hyperparameters found: " + best_hypers)
        logger.info("Achieved {} with this hyperparameters".format(best_recall))
        logger.info("-----------------------------------------------------\n")
    def execute(self, train_params, eval_params):
        since = time.time()
        best_recall_iter = 0
        scores = list()
        #self.comp_list = list()
        #self.comp_list_mean = list()
        for e in range(1, train_params['num_epochs'] + 1):
            if 'test' in self.config['mode'].split('_'):
                best_recall_iter = self.evaluate(eval_params, scores, 0, 0)
            # If not testing
            else:
                logger.info('Epoch {}/{}'.format(e, train_params['num_epochs']))
                if e == 31:
                    logger.info("reduce learning rate")
                    self.encoder.load_state_dict(torch.load(
                        osp.join(self.save_folder_nets, self.fn + '.pth')))
                    self.gnn.load_state_dict(torch.load(osp.join(self.save_folder_nets,
                        'gnn_' + self.fn + '.pth')))
                    for g in self.opt.param_groups:
                        g['lr'] = train_params['lr'] / 10.

                if e == 51:
                    logger.info("reduce learning rate")
                    self.gnn.load_state_dict(torch.load(osp.join(self.save_folder_nets,
                        'gnn_' + self.fn + '.pth')))

                    self.encoder.load_state_dict(torch.load(
                        osp.join(self.save_folder_nets, self.fn + '.pth')))
                    for g in self.opt.param_groups:
                        g['lr'] = train_params['lr'] / 10.

                # Normal training with backpropagation
                for x, Y, I, P in tqdm(self.dl_tr):
                    loss = self.forward_pass(x, Y, I, P, train_params)
                    # Check possible net divergence
                    if torch.isnan(loss):
                        logger.error("We have NaN numbers, closing\n\n\n")
                        return 0.0, self.encoder

                    # Backpropagation
                    if train_params['is_apex'] == 1:
                        with amp.scale_loss(loss, self.opt) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    
                    self.opt.step()
                        
                    if self.center:
                        for param in self.center.parameters():
                            param.grad.data *= (
                                    1. / self.config['train_params']['loss_fn']['scaling_center'])
                        self.opt_center.step()

                best_recall_iter = self.evaluate(eval_params, scores, e, best_recall_iter)

            # compute epoch loss mean
            [self.losses_mean[k].append(sum(v) / len(v)) for k, v in self.losses.items()]
            #self.comp_list_mean.append(sum(self.comp_list)/ len(self.comp_list))
            losses = defaultdict(list)
            logger.info('Loss Values: ')
            logger.info(', '.join([str(k) + ': ' + str(v[-1]) for k, v in self.losses_mean.items()]))

        end = time.time()

        self.save_results(train_params, since, end, best_recall_iter, scores)

        return best_recall_iter, self.encoder

    def forward_pass(self, x, Y, I, P, train_params):
        Y = Y.to(self.device)
        self.opt.zero_grad()
        if self.center:
            self.opt_center.zero_grad()
        
        probs, fc7 = self.encoder(x.to(self.device), output_option=train_params['output_train_enc'])
 
        # Compute CE Loss
        loss = 0
        if self.ce:
            loss0 = self.ce(probs/self.config['train_params']['temperatur'], Y)
            loss+= train_params['loss_fn']['scaling_ce'] * loss0
            self.losses['Cross Entropy'].append(loss.item())

        # Add other losses of not pretraining
        if self.gnn_loss or self.of:
            edge_attr, edge_index, fc7 = self.graph_generator.get_graph(fc7, Y)
            if type(loss) != int:
                loss = loss.cuda(self.device)
            pred, feats = self.gnn(fc7, edge_index, edge_attr, train_params['output_train_gnn'])
        
        #self.comp_list.append(self.comp(fc7, feats[-1]))

        if self.gnn_loss:
            if self.every:
                loss1 = [gnn_loss(pr/self.config['train_params']['temperatur'], Y) for gnn_loss, pr in zip(self.gnn_loss, pred)]
            else:
                loss1 = [self.gnn_loss(pred[-1]/self.config['train_params']['temperatur'], Y)]
            lo = [train_params['loss_fn']['scaling_gnn'] * l for l in loss1]
            loss += sum(lo)
            [self.losses['GNN'+ str(i)].append(l.item()) for i, l in enumerate(loss1)]

        # Compute center loss
        if self.center:
            loss2 = self.center(feats[-1], Y)
            loss += train_params['loss_fn']['scaling_center'] * loss2
            self.losses['Center'].append(loss2.item())

        # Compute Triplet Loss
        if self.triplet:
            triploss, _ = self.triplet(fc7, Y)
            loss += train_params['loss_fn']['scaling_triplet'] * triploss
            self.losses['Triplet'].append(triploss.item())

        # Compute MSE regularization
        if self.of:
            p = feats[-1].detach()[:-k] if k != 0 else feats[0].detach()
            of_reg = self.of(fc7, p)
            loss += train_params['loss_fn']['scaling_of'] * of_reg
            self.losses['OF'].append(of_reg.item())

        # Compute CE loss with soft targets = predictions of gnn
        if self.distill:
            target = torch.stack([self.soft_targets[p] for p in P]).to(self.device)
            distill = self.distill(probs/self.config['train_params']['loss_fn']['soft_temp'], target)
            loss += train_params['loss_fn']['scaling_distill'] * distill
            self.losses['Distillation'].append(distill.item())

        # compute MSE loss with feature vectors from gnn
        if self.of_pre:
            target = torch.stack([torch.tensor(self.feat_targets[p]) for p in P]).to(self.device)
            of_pre = self.of_pre(fc7, target)

            loss += train_params['loss_fn']['scaling_of_pre'] * of_pre
            self.losses['OF Pretrained'].append(of_pre.item())
        
        # compute relaional teacher-student loss
        if self.distance:
            teacher = torch.stack([torch.tensor(self.feat_targets[p]) for p in P]).to(self.device)
            dist = self.distance(teacher, fc7)

            loss += train_params['loss_fn']['scaling_distance'] * dist
            self.losses['Distance'].append(dist.item())

        self.losses['Total Loss'].append(loss.item())

        return loss

    def evaluate(self, eval_params, scores, e, best_recall_iter):
        if not self.config['mode'] == 'pretraining':
            with torch.no_grad():
                logger.info('EVALUATION')
                if self.config['mode'] == 'train' or self.config['mode'] == 'test':
                    mAP, top = self.evaluator.evaluate(self.encoder, self.dl_ev,
                            self.gallery_dl, net_type=self.net_type,
                            dataroot=self.config['dataset']['dataset_short'], 
                            nb_classes=self.config['dataset']['num_classes'])
                else: # all other modes that involve gnn during test time  
                    mAP, top = self.evaluator.evaluate(self.encoder, self.dl_ev,
                            self.gallery_dl, self.gnn, self.graph_generator, 
                            dl_ev_gnn=self.dl_ev_gnn, net_type=self.net_type,
                            dataroot=self.config['dataset']['dataset_short'],
                            nb_classes=self.config['dataset']['num_classes'])          
                
                scores.append((mAP, top))
                recall = top[0]

                self.encoder.current_epoch = e
                if recall > best_recall_iter:
                    best_recall_iter = recall 
                    if 'test' not in self.config['mode'].split('_'):
                        torch.save(self.encoder.state_dict(),
                                   osp.join(self.save_folder_nets,
                                            self.fn + '.pth'))
                        torch.save(self.gnn.state_dict(), 
                                osp.join(self.save_folder_nets,
                                            'gnn_' + self.fn + '.pth'))

        else:
            logger.info(
                'Loss {}, Accuracy {}'.format(torch.mean(loss.cpu()),
                                              self.running_corrects / self.denom))

            scores.append(self.running_corrects / self.denom)
            self.denom = 0
            self.running_corrects = 0
            if scores[-1] > best_recall_iter:
                best_recall_iter = scores[-1]
                torch.save(self.encoder.state_dict(),
                           osp.join(self.save_folder_nets,
                                    self.fn + '.pth'))

        return best_recall_iter


    def save_results(self, train_params, since, end, best_recall_iter, scores):
        logger.info(
            'Completed {} epochs in {}s on {}'.format(
                train_params['num_epochs'],
                end - since,
                self.dataset_short))

        file_name = str(
            best_recall_iter) + '_' + self.dataset_short + '_' + str(self.timer)
        if 'test' in self.config['mode'].split('_'):
            file_name = 'test_' + file_name
        
        results_dir = osp.join(self.save_folder_results, file_name)
        utils.make_dir(results_dir)

        if not self.config['mode'] == 'pretraining':
            with open(
                    osp.join(results_dir, file_name + '.txt'),
                    'w') as fp:
                fp.write(file_name + "\n")
                fp.write(str(self.config))
                fp.write('\n')
                fp.write('\n'.join('%s %s' % x for x in scores))
                fp.write("\n\n\n")

        # plot losses and scores
        if not 'test' in self.config['mode'].split('_'):
            for k, v in self.losses_mean.items():
                eps = list(range(len(v)))
                plt.plot(eps, v)
                plt.xlim(left=0)
                plt.ylim(bottom=0, top=14)
                plt.xlabel('Epochs')
            plt.legend(self.losses_mean.keys(), loc='upper right')
            plt.grid(True)
            plt.savefig(osp.join(results_dir, k + '.png'))
            plt.close()

            scores = [[score[0], score[1][0], score[1][1], score[1][2]] for score
                      in scores]
            for i, name in enumerate(['mAP', 'rank-1', 'rank-5', 'rank-10']):
                sc = [s[i] for s in scores]
                eps = list(range(len(sc)))
                plt.plot(eps, sc)
                plt.xlim(left=0)
                plt.ylim(bottom=0)
                plt.xlabel('Epochs')
                plt.ylabel(name)
                plt.grid(True)
                plt.savefig(osp.join(results_dir, name + '.png'))
                plt.close()
            '''
            plt.plot(len(self.comp_list_mean), self.comp_list_mean)
            plt.xlim(left=0)
            plt.ylim(bottom=0)
            plt.xlabel('Epochs')
            plt.ylabel('KL-Divergence')
            plt.grid(True)
            plt.savefig(osp.join(results_dir, 'KL.png'))
            plt.close()
            '''

    def get_loss_fn(self, params, num_classes):
        self.losses = defaultdict(list)
        self.losses_mean = defaultdict(list)
        self.every = self.config['models']['gnn_params']['every']
        
        # GNN loss
        if not self.every: # normal GNN loss
            if 'gnn' in params['fns'].split('_'):
                self.gnn_loss = nn.CrossEntropyLoss().to(self.device)
            elif 'lsgnn' in params['fns'].split('_'):
                self.gnn_loss = losses.CrossEntropyLabelSmooth(
                    num_classes=num_classes, dev=self.device).to(self.device)
            elif 'focalgnn' in params['fns'].split('_'):
                self.gnn_loss = losses.FocalLoss().to(self.device)
            else:
                self.gnn_loss = None
        
        else: # GNN loss after every layer
            if 'gnn' in params['fns'].split('_'):
                self.gnn_loss = [nn.CrossEntropyLoss().to(self.device) for
                        _ in range(self.config['models']['gnn_params']['gnn']['num_layers'])] 
            elif 'lsgnn' in params['fns'].split('_'):
                use_gpu = False if self.device == torch.device('cpu') else True
                self.gnn_loss = [losses.CrossEntropyLabelSmooth(
                    num_classes=num_classes, dev=self.gnn_dev, use_gpu=use_gpu).to(self.device) for
                        _ in range(self.config['models']['gnn_params']['gnn']['num_layers'])]

        # CrossEntropy Loss
        if 'lsce' in params['fns'].split('_'):
            use_gpu = False if self.device == torch.device('cpu') else True
            self.ce = losses.CrossEntropyLabelSmooth(
                num_classes=num_classes, dev=self.device, use_gpu=use_gpu).to(self.device)
        elif 'focalce' in params['fns'].split('_'):
            self.ce = losses.FocalLoss().to(self.device)
        elif 'ce' in params['fns'].split('_'):
            self.ce = nn.CrossEntropyLoss().to(self.device)
        else:
            self.ce = None

        # Center Loss
        if 'center' in params['fns'].split('_'):
            self.center = losses.CenterLoss(num_classes=num_classes).to(self.device)
            self.opt_center = torch.optim.SGD(self.center.parameters(),
                                              lr=0.5)
        else:
            self.center = None

        # triplet loss
        if 'triplet' in params['fns'].split('_'):
            self.triplet = losses.TripletLoss(margin=0.5).to(self.device)
        else:
            self.triplet = None
        
        # OF Loss
        if 'of' in params['fns'].split('_'):
            self.of = nn.MSELoss().to(self.device)
        else:
            self.of = None
        
        # Knowledge distillation
        if 'distillSh' in params['fns'].split('_'):
            self.distill = losses.CrossEntropyDistill().to(self.device)
            with open(params['preds'], 'r') as f:
                self.soft_targets = json.load(f)
            self.soft_targets = {k: F.softmax(torch.tensor(v)/params['soft_temp']) for k, v in self.soft_targets.items()}
        elif 'distillKL' in params['fns'].split('_'):
            self.distill = losses.KLDivWithLogSM().to(self.device)
            with open(params['preds'], 'r') as f:
                self.soft_targets = json.load(f)
            self.soft_targets = {k: F.softmax(torch.tensor(v)/params['soft_temp']) for k, v in self.soft_targets.items()}
        else:
            self.distill = None
        
        # Node feature distillation
        if 'ofpre' in params['fns'].split('_'):
            self.of_pre = nn.SmoothL1Loss().to(self.device)
            with open(params['feats'], 'r') as f:
                self.feat_targets = json.load(f)
        else:
            self.of_pre = None
        
        # Relational Loss
        if 'distance' in params['fns'].split('_'):
            self.distance = losses.DistanceLoss().to(self.device)
            with open(params['feats'], 'r') as f:
                self.feat_targets = json.load(f)
        else:
            self.distance = None
        
        #self.comp = nn.KLDivLoss()

    def update_params(self):
        self.sample_hypers() if 'hyper' in self.config['mode'] else None

        if 'test' in self.config['mode'].split('_'):
            self.config['train_params']['num_epochs'] = 1

    def sample_hypers(self):
        config = {'lr': 10 ** random.uniform(-5, -3),
                  'weight_decay': 10 ** random.uniform(-15, -6),
                  'num_classes_iter': random.randint(6, 15), #100
                  'num_elements_class': random.randint(3, 9),
                  'temperatur': random.random(),
                  'num_epochs': 40}
        self.config['train_params'].update(config)
        
        config = {'num_layers': random.randint(1, 4)}
        config = {'num_heads': random.choice([1, 2, 4, 8])}
        
        self.config['models']['gnn_params']['gnn'].update(config)

        logger.info("Updated Hyperparameters:")
        logger.info(self.config)

    def get_data(self, config, train_params, mode):
        loaders = data_utility.create_loaders(
                data_root=config['dataset_path'],
                num_workers=config['nb_workers'],
                num_classes_iter=train_params['num_classes_iter'],
                num_elements_class=train_params['num_elements_class'],
                trans=config['trans'],
                num_classes=self.config['dataset']['num_classes'], 
                net_type=self.net_type,
                bssampling=self.config['dataset']['bssampling'],
                mode=mode)
        
        self.dl_tr, self.dl_ev, self.gallery_dl, self.dl_ev_gnn = loaders
