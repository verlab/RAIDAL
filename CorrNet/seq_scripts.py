import os
import pdb
import sys
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from evaluation.slr_eval.wer_calculation import evaluate
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

def seq_train(loader, model, optimizer, device, epoch_idx, recoder, scaler):
    model.train()

    loss_value      = []

    for batch_idx, data in enumerate(tqdm(loader)):
        clr = [group['lr'] for group in optimizer.optimizer.param_groups]

        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        optimizer.zero_grad()

        with autocast():
            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)
            loss     = model.criterion_calculation(ret_dict, label, label_lgt)
        
        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print('loss is nan')
            #print(data[-1])
            print(str(data[1])+'  frames')
            print(str(data[3])+'  glosses')
            continue
        
        scaler.scale(loss).backward()
        scaler.step(optimizer.optimizer)
        scaler.update()

        loss_value.append(loss.item())

        if batch_idx % recoder.log_interval == 0:
            recoder.print_log(
                '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}'
                    .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[0]))
        del ret_dict
        del loss
    
    optimizer.scheduler.step() 
    recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))

    return 


def seq_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder,
             evaluate_tool="python"):
    model.eval()
    total_sent = []
    total_info = []
    total_conv_sent = []
    stat = {i: [0, 0] for i in range(len(loader.dataset.dict))}
    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)

        total_info += [file_name.split("|")[0] for file_name in data[-1]]
        total_sent += ret_dict['recognized_sents']
        total_conv_sent += ret_dict['conv_sents']
    try:
        python_eval = True if evaluate_tool == "python" else False
        write2file(work_dir + "output-hypothesis-{}.ctm".format(mode), total_info, total_sent)
        write2file(work_dir + "output-hypothesis-{}-conv.ctm".format(mode), total_info,
                   total_conv_sent)
        conv_ret = evaluate(
            prefix=work_dir, mode=mode, output_file="output-hypothesis-{}-conv.ctm".format(mode),
            evaluate_dir=cfg.dataset_info['evaluation_dir'],
            evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
            output_dir="epoch_{}_result/".format(epoch),
            python_evaluate=python_eval,
        )
        lstm_ret = evaluate(
            prefix=work_dir, mode=mode, output_file="output-hypothesis-{}.ctm".format(mode),
            evaluate_dir=cfg.dataset_info['evaluation_dir'],
            evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
            output_dir="epoch_{}_result/".format(epoch),
            python_evaluate=python_eval,
            triplet=True,
        )
    except:
        print("Unexpected error:", sys.exc_info()[0])
        lstm_ret = 100.0
    finally:
        pass
    
    recoder.print_log(f"Epoch {epoch}, {mode} {lstm_ret: 2.2f}%", f"{work_dir}/{mode}.txt")
    return lstm_ret


def setMCDropout(model, enabled=True):
    if enabled:
        model.eval()
        for m in model.modules():
            if 'Dropout' in m.__class__.__name__:
                m.train()

            elif isinstance(m, (nn.RNNBase)):
                # This finds nn.LSTM, nn.GRU, and nn.RNN modules
                # We check if it *has* dropout and if it's > 0
                if hasattr(m, 'dropout') and m.dropout > 0:
                    m.train()

    else:
        model.eval()


def seq_feature_generation(loader, model, device, mode, work_dir, recoder, baldIterations=0):
    model.eval()

    src_path = os.path.abspath(f"{work_dir}{mode}")


    # Remove ./features so I can run AL experiments in parallel.
    '''
    tgt_path = os.path.abspath(f"./features/{mode}")
    if not os.path.exists("./features/"):
        os.makedirs("./features/")

    if os.path.islink(tgt_path):
        curr_path = os.readlink(tgt_path)
        if work_dir[1:] in curr_path and os.path.isabs(curr_path):
            return
        else:
            os.unlink(tgt_path)
    else:
        if os.path.exists(src_path) and len(loader.dataset) == len(os.listdir(src_path)):
            os.symlink(src_path, tgt_path)
            return
    '''

    for batch_idx, data in tqdm(enumerate(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt)

            ctcDecodedSeqsWithDropout     = []
            classifierFeaturesWithDropout = []


            # bilstm_features are the embeddings that are output by the BiLSTM module
            biLSTMFeaturesNoDropout     = ret_dict["bilstm_features"]

            # classifier_features are the not-softmaxxed logits output by the classifier. They are the same inputs that the
            # ctc decoder receives.
            classifierFeaturesNoDropout = ret_dict["classifier_features"]


            ctcTimestepsNoDropout    = ret_dict["ctcBeamTimesteps"]
            ctcDecodedSeqsNoDropout  = ret_dict["ctcDecodedSeqs"]

            if baldIterations > 0:
                setMCDropout(model, enabled=True)


                # Run N iterations with dropout enabled
                for i in range(baldIterations):
                    modelOutput = model(vid, vid_lgt)

                    classifierFeaturesWithDropout.append(modelOutput["classifier_features"])
                    ctcDecodedSeqsWithDropout.append(modelOutput["ctcDecodedSeqs"])

                # Disable dropout for next iteration
                setMCDropout(model, enabled=False)


        if not os.path.exists(src_path):
            os.makedirs(src_path)
        start = 0
        for sample_idx in range(len(vid)):
            end = start + data[3][sample_idx]
            filename = f"{src_path}/{data[-1][sample_idx].split('|')[0]}_features.npy"
            save_file = {
                "label": data[2][start:end],

                "biLSTMFeaturesNoDropout":   biLSTMFeaturesNoDropout,
                
                "classifierFeaturesNoDropout":   classifierFeaturesNoDropout,
                "classifierFeaturesWithDropout": classifierFeaturesWithDropout,

                "ctcTimestepsNoDropout":   ctcTimestepsNoDropout,

                "ctcDecodedSeqsNoDropout":   ctcDecodedSeqsNoDropout,
                "ctcDecodedSeqsWithDropout": ctcDecodedSeqsWithDropout
            }
            np.save(filename, save_file)
            start = end
        assert end == len(data[2])
    #os.symlink(src_path, tgt_path)


def write2file(path, info, output):
    filereader = open(path, "w")
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                "{} 1 {:.2f} {:.2f} {}\n".format(info[sample_idx],
                                                 word_idx * 1.0 / 100,
                                                 (word_idx + 1) * 1.0 / 100,
                                                 word[0]))
