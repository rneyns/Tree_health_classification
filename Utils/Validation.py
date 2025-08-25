import sklearn
import torch.nn as nn
import numpy as np


def class_wise_acc(y_pred, y_test, num_classes):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    acc_classwise = []
    total_correct = []
    total_val_batch = []

    for i in range(num_classes):
        correct_pred_class = correct_pred[y_test == i]
        acc_class = correct_pred_class.sum() / len(correct_pred_class)
        acc_classwise.append(acc_class)
        total_correct.append(correct_pred_class.sum().cpu().data.numpy())
        # total_val_batch.append(torch.tensor(len(correct_pred_class), dtype=torch.int8))
        total_val_batch.append(len(correct_pred_class))
        # print('Accuracy of {} : {} / {} = {:.4f} %'.format(i, correct_pred_class.sum() , len(correct_pred_class) , 100 * acc_class))

    return acc_classwise, total_correct, total_val_batch


def valid(args, model, device, dataloader):
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=1)

    _loss = 0
    _loss_a = 0
    _loss_v = 0
    _out = []
    _label = []
    counter = 0
    total_correct_sum = np.zeros(args.numClasses)
    total_val_batch_sum = np.zeros(args.numClasses)

    if args.dataset == 'VGGSound':
        n_classes = 309
    elif args.dataset == 'KineticSound':
        n_classes = 31
    elif args.dataset == 'CREMAD':
        n_classes = 6
    elif args.dataset == 'AVE':
        n_classes = 28
    elif args.dataset == 'CGMNIST':
        n_classes = 10
    elif args.dataset == 'TREES':
        n_classes = args.numClasses

    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        model.eval()
        # TODO: more flexible
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]
        acc_a = [0.0 for _ in range(n_classes)]
        acc_v = [0.0 for _ in range(n_classes)]

        for step, (image, x_categ, x_cont, label, cat_mask, con_mask) in enumerate(dataloader):

            x_categ = x_categ.to(device)  # B x 4 x 12
            x_cont = x_cont.to(device)  # B x 4 x 12
            cat_mask = cat_mask.to(device)  # B x 4 x 12
            con_mask = con_mask.to(device)  # B x 4 x 12
            image = image.to(device)  # B x 1(image count) x 3 x 224 x 224
            label = label.to(device)

            _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, model.tab_net,
                                                         vision_dset=True)

            a, v, out = model(image, x_categ_enc, x_cont_enc)  # gray colored

            loss = criterion(out, label)
            loss_v = criterion(v, label)
            loss_a = criterion(a, label)

            # Calculating the class-wise accuracy
            _, total_correct, total_val_batch = class_wise_acc(out, label, num_classes=args.numClasses)
            # Calculating the class-wise accuracy
            _, total_correct, total_val_batch = class_wise_acc(out, label, num_classes=args.numClasses)
            # Calculating the class-wise accuracy
            _, total_correct, total_val_batch = class_wise_acc(out, label, num_classes=args.numClasses)

            total_correct_sum = np.sum([total_correct_sum, total_correct], axis=0)
            total_val_batch_sum = np.sum([total_val_batch_sum, total_val_batch], axis=0)

            _loss += loss.item()
            _loss_a += loss_a.item()
            _loss_v += loss_v.item()
            counter += 1

            if args.fusion_method == 'sum':
                out_v = (torch.mm(v, torch.transpose(model.fusion_module.ffc_y.weight, 0, 1)) +
                         model.fusion_module.ffc_y.bias)
                out_a = (torch.mm(a, torch.transpose(model.fusion_module.ffc_x.weight, 0, 1)) +
                         model.fusion_module.ffc_x.bias)
            elif args.fusion_method == 'concat':
                weight_size = model.fusion_module.ffc_out.weight.size(1)
                out_v = (torch.mm(v, torch.transpose(model.fusion_module.ffc_out.weight[:, weight_size // 2:], 0, 1))
                         + model.fusion_module.ffc_out.bias / 2)
                out_a = (torch.mm(a, torch.transpose(model.fusion_module.ffc_out.weight[:, :weight_size // 2], 0, 1))
                         + model.fusion_module.ffc_out.bias / 2)
            elif args.fusion_method == 'film' or args.fusion_method == 'gated':
                out_v = out
                out_a = out

            prediction = softmax(out)
            pred_v = softmax(out_v)
            pred_a = softmax(out_a)

            for i in range(out.shape[0]):

                ma = np.argmax(prediction[i].cpu().data.numpy())
                v = np.argmax(pred_v[i].cpu().data.numpy())
                a = np.argmax(pred_a[i].cpu().data.numpy())
                num[label[i]] += 1.0
                _out.append(int(ma))
                _label.append(int(label[i]))

                # pdb.set_trace()
                if np.asarray(label[i].cpu()) == ma:
                    acc[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == v:
                    acc_v[label[i]] += 1.0
                if np.asarray(label[i].cpu()) == a:
                    acc_a[label[i]] += 1.0

        # printing the class_wise_accuracies
        class_accuracies = []
        class_accuracies_tab = []
        class_accuracies_img = []
        for i in range(args.numClasses):
            class_accuracies.append(total_correct_sum[i] / total_val_batch_sum[i])

        # calculating the kappa value
        # print(np.array(_out),np.array(_label))
        kappa = sklearn.metrics.cohen_kappa_score(np.array(_out), np.array(_label))
        F1 = sklearn.metrics.f1_score(np.array(_out), np.array(_label), average='micro')
        precision_per_class = precision_score(np.array(_label), np.array(_out), average=None)
        recall_per_class = recall_score(np.array(_label), np.array(_out), average=None)

        # print(sklearn.metrics.confusion_matrix(np.array(_out),np.array(_label)))

        log = {"loss_val": float(_loss) / counter, "loss_img_val": float(_loss_a) / counter,
               "loss_tab_val": float(loss_v) / counter, "kappa": kappa, 'F1': F1}
        for i in range(args.numClasses):
            log.update({f"class{i}_precision": precision_per_class[i]})
            log.update({f"class{i}_recall": recall_per_class[i]})

        wandb.log(log)

    return sum(acc) / sum(num), sum(acc_a) / sum(num), sum(acc_v) / sum(num)


