
import logging
import os
from abc import abstractmethod

import torch
from numpy import inf
import time
import pandas as pd


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # File handler to log messages to a file
        file_handler = logging.FileHandler('training_log_RCCL.txt')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
        self.logger.addHandler(file_handler)


        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir
        self.record_dir = args.record_dir


        self.caption_dir = os.path.join(self.record_dir, 'captions')
        if not os.path.exists(self.caption_dir):
            os.makedirs(self.caption_dir)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError


    def _save_captions_to_txt(self, epoch, split, captions_data):

        txt_filename = f"captions_epoch_{epoch}_{split}.txt"
        txt_path = os.path.join(self.caption_dir, txt_filename)

        # Write header
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"=== Epoch {epoch} - {split.upper()} Set Caption Results ===\n")
            f.write(f"Generated at: {time.ctime()}\n")
            f.write(f"Total samples: {len(captions_data)}\n\n")


            for idx, data in enumerate(captions_data, 1):
                f.write(f"Sample {idx}\n")
                f.write(f"Image ID: {data['image_id']}\n")
                f.write(f"Ground Truth: {data['ground_truth']}\n")
                f.write(f"Prediction: {data['prediction']}\n")
                f.write("-" * 80 + "\n\n")

        self.logger.info(f"Saved {split} set captions to: {txt_path}")

    def train(self):
        not_improved_count = 0
        best_epoch = 0
        df = None
        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir)
        filename = os.path.join(self.record_dir, 'RCCL.csv')
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)


            log = {'epoch': epoch}
            log.update(result)
            if df is None:
                df = pd.DataFrame.from_dict(log, orient='index').T
            else:
                df = pd.concat([df, pd.DataFrame.from_records([log])], ignore_index=True)
            df.to_csv(filename, index=False)
            self._record_best(log)

            # Print logs
            for key, value in log.items():
                self.logger.info('\t{:15s}: {}'.format(str(key), value))

            # Monitor validation performance
            best = False
            if self.mnt_mode != 'off':
                try:
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning(f"Metric '{self.mnt_metric}' not found. Monitoring disabled.")
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                    best_epoch = epoch
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(f"Validation performance didn't improve for {self.early_stop} epochs. Stopping.")
                    break

            # Save checkpoint
            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
            print('Best performance in epoch: ', best_epoch)

        self._print_best()
        self._print_best_to_file()

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        self.logger.info(f'Best results (w.r.t {self.args.monitor_metric}) in validation set:')
        for key, value in self.best_recorder['val'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

        self.logger.info(f'Best results (w.r.t {self.args.monitor_metric}) in test set:')
        for key, value in self.best_recorder['test'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name + '.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = pd.concat([record_table, pd.DataFrame([self.best_recorder['val']])], ignore_index=True)
        record_table = pd.concat([record_table, pd.DataFrame([self.best_recorder['test']])], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("No GPU available. Training on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(f"Requested {n_gpu_use} GPUs, but only {n_gpu} available.")
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        self.logger.info(f"Saving checkpoint: {filename}")
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving best model: model_best.pth")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger.info(f"Checkpoint loaded. Resuming from epoch {self.start_epoch}")


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def save_result(self, filename, epoch, *context):
        with open(filename, 'a') as f:
            seq = '\t'
            filen = filename.split('/')[-1]
            if filen == 'train.txt':
                if epoch == 1:
                    f.write(f'epoch{seq}loss{seq}\n')
                resultrow = ''
                for con in context:
                    resultrow = resultrow + str(con) + seq
                f.write(f'{resultrow}\n')
            else:
                if epoch == 1:
                    metricsName = context[0].keys()
                    metricsNameStr = ''
                    for name in metricsName:
                        metricsNameStr = metricsNameStr + str(name) + seq
                    f.write(f'epoch{seq}{metricsNameStr}\n')
                resultrow = ''
                for con in context[0].values():
                    resultrow = resultrow + str(con) + seq
                f.write(f'{epoch}{seq}{resultrow}\n')

    def _train_epoch(self, epoch):
        self.logger.info('[{}/{}] Start training in training set.'.format(epoch, self.epochs))
        train_loss = 0
        self.model.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                self.device)

            # Get model output (adjust if your model expects `labels_batch` here)
            output, clwps_loss = self.model(images, reports_ids, mode='train')
            loss = 0.9 * self.criterion(output, reports_ids, reports_masks) + 0.1 * clwps_loss



            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_idx % self.args.log_period == 0:
                avg_loss = train_loss / (batch_idx + 1)
                self.logger.info('[{}/{}] Step: {}/{}, Training Loss: {:.5f}'.format(
                    epoch, self.epochs, batch_idx, len(self.train_dataloader), avg_loss))

        log = {'train_loss': train_loss / len(self.train_dataloader)}


        self.logger.info('[{}/{}] Evaluating validation set.'.format(epoch, self.epochs))
        self.model.eval()
        val_captions_data = []
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)

                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                #print('reports:',reports) # For Report Generation in Console
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                #print('ground_truths:', ground_truths)  #For Report Generation in Console

                # Collect data for TXT (NEW)
                for img_id, gt, pred in zip(images_id, ground_truths, reports):
                    val_captions_data.append({
                        'image_id': img_id,
                        'ground_truth': gt.strip(),
                        'prediction': pred.strip()
                    })

                val_res.extend(reports)
                val_gts.extend(ground_truths)


            self._save_captions_to_txt(epoch, split='val', captions_data=val_captions_data)


            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})


        self.logger.info('[{}/{}] Evaluating test set.'.format(epoch, self.epochs))
        self.model.eval()
        test_captions_data = []
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)

                output, _ = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())  # Predictions
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())  # GT


                for img_id, gt, pred in zip(images_id, ground_truths, reports):
                    test_captions_data.append({
                        'image_id': img_id,
                        'ground_truth': gt.strip(),
                        'prediction': pred.strip()
                    })

                test_res.extend(reports)
                test_gts.extend(ground_truths)


            self._save_captions_to_txt(epoch, split='test', captions_data=test_captions_data)


            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)}, {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            self.save_result(self.checkpoint_dir + 'test.txt', epoch, test_met)
            test_res, test_gts = pd.DataFrame(test_res), pd.DataFrame(test_gts)
            test_res.to_csv(os.path.join(self.args.save_dir, "res.csv"), index=False, header=False) # Descriptive Results in CSV
            test_gts.to_csv(os.path.join(self.args.save_dir, "gts.csv"), index=False, header=False) # Descriptive Results in CSV
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        self.lr_scheduler.step()
        return log