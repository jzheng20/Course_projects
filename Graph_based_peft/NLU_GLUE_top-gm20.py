import torch

def print_memory_usage():
    # 已分配的显存
    allocated = torch.cuda.memory_allocated() / 1024**2  # 转换为 MB
    # 当前保留的显存
    reserved = torch.cuda.memory_reserved() / 1024**2   # 转换为 MB
    print(f"Allocated Memory: {allocated:.2f} MB")
    print(f"Reserved Memory: {reserved:.2f} MB")

import torch.nn as nn
import numpy as np
from args import *
import os
import time
import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig, 
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig, 
)
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm
from top_gm2 import utilized


def set_bias_trainable(model,module_name):
    for name, module in model.named_children():
        if name in module_name:
            for name0, param in module.named_parameters():
                if "bias" in name0:
                    param.requires_grad=True
                
            #return model
        else:
            set_bias_trainable(module,module_name)
            #result=get_nested_linear_iter(module)
           # if result is not None:
               # return result 
    return None

torch.cuda.empty_cache()

def get_nested_linear_iter(model,module_name):
    for name, module in model.named_children():
        if name in module_name:
            for name0, param in module.named_parameters():
                print(name)
                print(name0)
                print(param.requires_grad)
                
            #return model
        else:
            get_nested_linear_iter(module,module_name)
            #result=get_nested_linear_iter(module)
           # if result is not None:
               # return result 
    return None

args = get_args() 
print(args.weight_decay)
seeds=np.array([0]) 
for seed in seeds:
    args.seed=seed
    print(args.seed) 

    torch.manual_seed(args.seed)
    task = args.task
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    num_labels = 2
    if task == "stsb":
        num_labels = 1
    
 

    def log(*pargs):
        path_log = './logs_glue/' + task + '/' + args.model_name_or_path.split("-")[1] + '/bs' + str(args.bs) + 'maxlen' + str(args.max_length) + 'f_lr' + str(args.fft_lr)+ 'h_lr' + str(args.head_lr) + \
          'num' + str(args.n_frequency) + 'scale' + str(args.scale) + 'seed' + str(args.seed) + '.txt'
        print(path_log)
        with open(path_log, mode = 'a+') as w:
            w.write(" ".join(["{}".format(t) for t in pargs]))
            w.write("\n")

    if any(k in args.model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else: 
        padding_side = "right"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
 
    datasets = load_dataset("glue",task) 
    metric=load_metric("glue",task) 

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        if task == 'sst2' or task == 'cola':
            outputs = tokenizer(examples["sentence"], truncation=True, max_length=args.max_length)
        elif task == 'qnli':
            outputs = tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=args.max_length)
        elif task == 'qqp':
            outputs = tokenizer(examples["question1"], examples["question2"], truncation=True, max_length=args.max_length)
        else:
            outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=args.max_length)
        return outputs

    if task == 'sst2' or task == 'cola':
        tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence"],
        )
    elif task == 'qnli':
        tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "question", "sentence"],
        )
    elif task == 'qqp':
        tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "question1", "question2"],
        )
    else:
        tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2"],
        )

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")


    # Instantiate dataloaders.
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=args.bs)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=args.bs)

 
    for fft_lr in np.array([1e-3]): #1e-1,1e-2,1e-3
        for head_lr in np.array([5e-2]):#5e-2, 5e-3,5e-4,5e-5；0.005,0.0005
            for weight_decay in np.array([0]):#1e-1, 1e-2,5e-2, 5e-3,5e-4,0
               # args.fft_lr=fft_lr 
                #args.head_lr=head_lr
                #args.weight_decay= weight_decay
                
                print(args.head_lr)
                print(args.fft_lr)
                print(args.weight_decay)
                model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path,num_labels=num_labels,return_dict=True)
                from top_gm2 import topModel,utilized
                from utils import print_trainable_parameters 
                #from com import utilized
                for param in model.parameters():
                    param.requires_grad = False 
                #qkv_name=[['query','key','value','dense'],['dense'],['dense']]
 
                print("===================r=1==========================") 
                r=10
                module_name=['query','key','value']#['query','key','value','dense']
                model=topModel.Loading(args, model,r, module_name,device)
                #get_nested_linear_iter(model,module_name)
                #get_nested_linear_iter(model)
                #for param in model.parameters():
                  #   param.requires_grad = False 
                
                set_bias_trainable(model,["query","key","value","dense","classifier"])
                utilized.set_extra_trainable(model, ["classifier"]) 
                utilized.print_requires_grad(model)
                print(model)
                print_trainable_parameters(model)
                KK=0
                #for n,p in model.named_parameters():
                  #  if 
                # for param in model.parameters(): 
                #     if param.requires_grad == True:
                #         KK=1+KK
                total_KK=100
                print(total_KK)
                args.target_KK= 60
                args.importnodes_num= 30
                args.MODE_SA = "True" 
                #args.MODE_SA = "False"
                print(args.MODE_SA)
                #args.init_warmup=1
                #args.final_warmup=2
                args.mask_interval=1
                args.beta1=0.85
                args.beta2=0.85
  
                torch.cuda.empty_cache()

                head_param = list(map(id, model.classifier.parameters()))

                #others_param = filter(lambda p: id(p) not in head_param, model.parameters())
                others_param = filter(lambda p: p.requires_grad and id(p) not in head_param,model.parameters())
        
                optimizer = AdamW([{"params": model.classifier.parameters(), "lr": args.head_lr},{"params": others_param, "lr":args.fft_lr}],weight_decay=args.weight_decay)
                #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

                # Instantiate scheduler
                lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=0.06 * (len(train_dataloader) * args.num_epochs),
                    num_training_steps=(len(train_dataloader) * args.num_epochs),
                    )
                from top_gm2.asa_gm import SubspacesAllocator
                if args.MODE_SA == "True":
                    subspaces_allocator =  SubspacesAllocator(
                        model, 
                        target_KK=args.target_KK,
                        importnodes_num=args.importnodes_num,
                        init_warmup=args.init_warmup,
                        final_warmup=args.final_warmup,
                        mask_interval=args.mask_interval,
                        beta1=args.beta1,
                        beta2=args.beta2,
                        )
                else:
                    subspaces_allocator=None

                acc_list = []
                model.to(device) 
                max_steps=  args.num_epochs * len(train_dataloader)
                samp_num=0
                if subspaces_allocator is not None:
                    subspaces_allocator.set_total_step(args.num_epochs)
                for epoch in range(args.num_epochs):
                     model.train()
                     for step, batch in enumerate(tqdm(train_dataloader)): 
                         
                         batch.to(device)
                         outputs = model(**batch)
                         #loss_old = outputs.loss
                         #loss=LorTAloss.newloss_function(['roberta.encoder.lorta_att_L1_0','roberta.encoder.lorta_att_L2_0',
                                                         # 'roberta.encoder.lorta_att_L1_1','roberta.encoder.lorta_att_L2_1'],
                                                        # loss_old,model)
                         #loss.backward(retain_graph=True)
                         loss = outputs.loss
                         loss.backward()
                         #print_memory_usage()
                         if step == 0:
                                 print_memory_usage()
                         optimizer.step() 
                         #print_memory_usage()
                         if subspaces_allocator is not None:
                             if step == 0:
                                 #print_memory_usage() 
                                 curr_KK, is_collect, model =  subspaces_allocator.update_and_mask(model, total_KK, epoch) 
                                 subspaces_allocator.set_total_step(args.num_epochs)
                                 #print(chosen_points)
                         torch.cuda.empty_cache()
                         if step == 0 and subspaces_allocator is not None:
                             #print(1111111111)
                             #print(epoch)
                             print(curr_KK)
                             print(total_KK)
                         if  subspaces_allocator is not None:
                             if is_collect and samp_num < 250:
                                 total_KK=subspaces_allocator.collect_scores(model)
                         #if epoch>=1:
                           #  print(total_KK)
                         #subspaces_allocator.collect_scores(model)
                         #print(subspaces_allocator.score_point)
   
                        
                         lr_scheduler.step()
                         optimizer.zero_grad()
                         #torch.cuda.empty_cache()
                         #print_memory_usage()

                     model.eval()
                     for step, batch in enumerate(tqdm(eval_dataloader)):
                         batch.to(device)
                         with torch.no_grad():
                             outputs = model(**batch)
                         if task == "stsb":
                             predictions = outputs.logits
                         else:
                             predictions = outputs.logits.argmax(dim=-1)
                         predictions, references = predictions, batch["labels"] 
                         metric.add_batch(
                             predictions=predictions,
                             references=references,
                             )

                     eval_metric = metric.compute()
                     if task == "stsb":
                         acc_list.append(eval_metric['pearson'])
                         print(f"epoch {epoch}:", eval_metric, '\033[32m,current_best_pearson:\033[0m',max(acc_list),'train_loss:',loss)
                     elif task == 'cola':
                         acc_list.append(eval_metric['matthews_correlation'])
                         print(f"epoch {epoch}:", eval_metric, '\033[32m, current_best_corr:\033[0m',max(acc_list),'train_loss:',loss) 
                     else:
                         acc_list.append(eval_metric['accuracy'])
                         print(f"epoch {epoch}:", eval_metric, '\033[32m, current_best_acc:\033[0m',max(acc_list),'train_loss:',loss) 

        
        

