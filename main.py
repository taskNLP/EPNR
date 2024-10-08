import argparse
import json
import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import transformers
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader

import config
import data_loader
import utils
from model import Model


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        bert_params = set(self.model.bert.parameters())
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': config.learning_rate,
             'weight_decay': config.weight_decay},
        ]

        self.optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=config.warm_factor * updates_total,
                                                                      num_training_steps=updates_total)

    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        pred_result = []
        label_result = []

        for i, data_batch in enumerate(data_loader):
            data_batch = [data.cuda() for data in data_batch[:-2]]

            bert_inputs, grid_labels, grid_mask2d, entity_mask, pieces2word, pieces2entity, sent_length, entity_length = data_batch

            outputs = model(bert_inputs, pieces2word, pieces2entity, entity_mask, sent_length, grid_mask2d)

            loss = self.criterion(outputs[grid_mask2d], grid_labels[grid_mask2d])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.clip_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_list.append(loss.cpu().item())

            outputs = torch.argmax(outputs, -1)
            grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
            outputs = outputs[grid_mask2d].contiguous().view(-1)

            label_result.append(grid_labels.cpu())
            pred_result.append(outputs.cpu())

            self.scheduler.step()
        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)
        logger.info("loss={}".format(np.mean(loss_list)))
        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")

        table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall"])
        table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] +
                      ["{:3.4f}".format(x) for x in [f1, p, r]])
        logger.info("\n{}".format(table))
        return 0

    def eval(self, epoch, data_loader, is_test=False):
        self.model.eval()

        pred_result = []
        label_result = []

        total_r = 0
        total_p = 0
        total_c = 0
        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                relation_list = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch[:-2]]
                bert_inputs, grid_labels, grid_mask2d, entity_mask, pieces2word, pieces2entity, sent_length, entity_length = data_batch

                outputs = model(bert_inputs, pieces2word, pieces2entity, entity_mask, sent_length, grid_mask2d)

                outputs = torch.argmax(outputs, -1)
                c, p, r, _ = utils.decode(outputs.cpu().numpy(), relation_list, entity_length.cpu().numpy())

                total_r += r
                total_p += p
                total_c += c

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")
        logger.info("total_c = {}, total_p = {}, total_r = {}".format(total_c, total_p, total_r))
        r_f1, r_p, r_r = utils.cal_f1(total_c, total_p, total_r)

        title = "EVAL" if not is_test else "TEST"
        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Relation"] + ["{:3.4f}".format(x) for x in [r_f1, r_p, r_r]])

        logger.info("\n{}".format(table))
        return r_f1

    def predict(self, epoch, data_loader, data):
        self.model.eval()

        pred_result = []
        label_result = []

        result = []

        total_r = 0
        total_p = 0
        total_c = 0

        i = 0
        with torch.no_grad():
            for data_batch in data_loader:
                relation_list = data_batch[-1]
                entity_list = data_batch[-2]
                sentence_batch = data[i:i + config.batch_size]
                data_batch = [data.cuda() for data in data_batch[:-2]]
                bert_inputs, grid_labels, grid_mask2d, entity_mask, pieces2word, pieces2entity, sent_length, entity_length = data_batch

                outputs = model(bert_inputs, pieces2word, pieces2entity,entity_mask, sent_length, grid_mask2d)

                outputs = torch.argmax(outputs, -1)
                c, p, r, decode_result = utils.decode(outputs.cpu().numpy(), relation_list, entity_length.cpu().numpy())

                for relations, sentence, entity_li in zip(decode_result, sentence_batch, entity_list):
                    sentence_text = "".join(sentence["sentence"])
                    instance = {"text": sentence_text, "relation_of_mention": []}
                    for relation in relations:
                        instance["relation_of_mention"].append(utils.decode_relation(relation, entity_li, config.vocab, sentence_text))
                    result.append(instance)

                total_r += r
                total_p += p
                total_c += c

                grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
                outputs = outputs[grid_mask2d].contiguous().view(-1)

                label_result.append(grid_labels.cpu())
                pred_result.append(outputs.cpu())
                i += config.batch_size

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(),
                                                      pred_result.numpy(),
                                                      average="macro")
        r_f1, r_p, r_r = utils.cal_f1(total_c, total_p, total_r)
        print(total_c, total_p, total_r)

        title = "TEST"
        logger.info('{} Label F1 {}'.format("TEST", f1_score(label_result.numpy(),
                                                            pred_result.numpy(),
                                                            average=None)))

        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])
        table.add_row(["Relation"] + ["{:3.4f}".format(x) for x in [r_f1, r_p, r_r]])

        logger.info("\n{}".format(table))

        print(len(result))
        with open(config.predict_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

        return r_f1

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/CMedCausal.json')
    parser.add_argument('--save_path', type=str, default='./model.pt')
    parser.add_argument('--predict_path', type=str, default='./CMedCausal.json')
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--dist_emb_size', type=int)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--conv_hid_size', type=int)
    parser.add_argument('--bert_hid_size', type=int)

    parser.add_argument('--emb_dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--warm_factor', type=float)

    parser.add_argument('--use_bert_last_4_layers', type=int, help="1: true, 0: false")

    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    config = config.Config(args)

    logger = utils.get_logger(config.dataset)
    logger.info(config)
    config.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    # random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logger.info("Loading Data")
    datasets, ori_data = data_loader.load_data_bert(config)

    train_loader, dev_loader, test_loader = (
        DataLoader(dataset=dataset,
                   batch_size=config.batch_size,
                   collate_fn=data_loader.collate_fn,
                   shuffle=i == 0,
                   num_workers=4,
                   drop_last=i == 0)
        for i, dataset in enumerate(datasets)
    )

    updates_total = len(datasets[0]) // config.batch_size * config.epochs

    logger.info("Building Model")
    model = Model(config)

    model = model.cuda()

    trainer = Trainer(model)

    best_f1 = 0
    best_test_f1 = 0
    for i in range(config.epochs):
        logger.info("Epoch: {}".format(i))
        trainer.train(i, train_loader)
        f1 = trainer.eval(i, dev_loader)
        test_f1 = trainer.eval(i, test_loader, is_test=True)
        if best_f1 > f1:
            best_f1 = f1
            best_test_f1 = test_f1
            trainer.save(config.save_path)

    logger.info("Best TEST F1: {:3.4f}".format(best_f1))
    trainer.load(config.save_path)
    trainer.predict("Final", test_loader, ori_data[-1])