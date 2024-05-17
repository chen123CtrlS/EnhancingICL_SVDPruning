import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from my_utils.data_load import load_dataset_train_and_test
from my_utils.data_process_mc import sample_data_copa,cat_demo_copa
from my_utils.metrics import normalize_answer,calculate_metric
from transformers import AutoTokenizer
from transformers import GPTJForCausalLM
import torch
from tqdm import tqdm
from laser.log_utils import Logger
from laser.LaserWrapper import LaserWrapper
import warnings
import numpy as np
warnings.filterwarnings('ignore')

search = True
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
few_shot = True
# TODO set hyperparametric
save_dir = f"/home/EnhancingICL_SVDPruning/src/results/"
logger = Logger(save_dir=save_dir, fname=f"test.txt")
task_name  = 'copa'
local_data_path = r'' #TODO:your local dataset (if yes, otherwise '')
# choices=['k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_up', 'fc_out', 'None', 'dont']
l_name = "mlp"
# list(range(-1, 27))
l_num = 27
rates = [0,1.0,5.0,7.5,9.0,9.5,9.9,9.95]
llm_name = "GPTJ"
llm_path = "EleutherAI/gpt-j-6B"
save_path = r'/home/EnhancingICL_SVDPruning/src/results/gptj/'
txtname = llm_name+'_'+task_name+'_'+l_name+str(l_num)+'.txt'
demo_size = 5
eval_data_size = 100


# load data
dataset = load_dataset_train_and_test(task_name=task_name,local_data_path=local_data_path)
# process data
demo_sample,eval_sample,_ = sample_data_copa(dataset['train'],seed=66, actual_sample_size = eval_data_size, demo_size = demo_size)
eval_data = cat_demo_copa(demo_sample,eval_sample,task_name, demo_shot=demo_size)
test_data = cat_demo_copa(demo_sample,dataset['test'],task_name, demo_shot=demo_size)
print('finish loading and processing ## {} ## data!'.format(task_name))
if search == False:
    run_test_number = min(len(test_data),50)
    eval_data=eval_data[:run_test_number]
    test_data=test_data[:run_test_number]
# load model
tokenizer = AutoTokenizer.from_pretrained(llm_path)
model = GPTJForCausalLM.from_pretrained(
        llm_path,
        revision="float16",
        torch_dtype=torch.float16
    )

def test(model,test_data):
    model = model.to(device)
    pairs = []
    for i,item in enumerate(tqdm(test_data)):
        premise = item['premise']
        question = item['question']
        choice1 = item['choice1']
        choice2 = item['choice2']
        label = item['label']
        if few_shot:
            demo_shot = item['demo_sentence']
        else:
            demo_shot= ''
        prompt1 = f"{premise}\n{question}\n{choice1}\n"
        prompt2 = f"{premise}\n{question}\n{choice2}\n"
        inputs1 = tokenizer(demo_shot+prompt1, return_tensors="pt").to(device)
        inputs2 = tokenizer(demo_shot+prompt2, return_tensors="pt").to(device)
        scores = []
        with torch.no_grad():
            results = model(inputs1.input_ids)
            scores.append(results.logits.mean().item())
            results = model(inputs2.input_ids)
            scores.append(results.logits.mean().item())
            pred = np.argmax(scores)
        # print((pred,label))
        pairs.append((pred,label))
        if (i+1)%1000==0:
            print('test data number {}  accuracy {}'.format(i+1,calculate_metric(pairs, 'accuracy')))
    acc = calculate_metric(pairs, 'accuracy')
    return acc,pairs
if search:
    f1_s = []
    with open(save_path+txtname, 'w', encoding='utf-8') as f:
        for rate in rates:
            print(l_name+' '+str(l_num)+' '+str(rate)+' ')
            f.write(l_name+' '+str(l_num)+' '+str(rate)+' ')
            model_edit = LaserWrapper.get_edited_model(model=model,
                                                                    lname=l_name,
                                                                    lnum= l_num,
                                                                    rate=rate,
                                                                    intervention="rank-reduction",
                                                                    logger=logger,
                                                                    in_place=True)
            f1,pairs = test(model_edit,eval_data)
            print(f1)
            f1_s.append(f1)
            f.write(str(f1)+'\n')

        best_clip_rate_id = torch.argmax(torch.tensor(f1_s)).item()
        best_clip_rate = rates[best_clip_rate_id]
        f.write("best_clip_rate: {}\n".format(best_clip_rate))
    del model_edit
else:
    best_clip_rate = 9.95
print('best_clip_rate'+': '+str(best_clip_rate))
# clear cache
del model 
torch.cuda.empty_cache()
# reload weight
model = GPTJForCausalLM.from_pretrained(
        llm_path,
        revision="float16",
        torch_dtype=torch.float16
    )
f1_test,_ = test(model,test_data)
model_edit = LaserWrapper.get_edited_model(model=model,
                                                                lname = l_name,
                                                                lnum = l_num,
                                                                rate = best_clip_rate,
                                                                intervention="rank-reduction",
                                                                logger=logger,
                                                                in_place=True)
f1_test_svd,_ = test(model_edit,test_data) 
print('test f1:{} -> {}'.format(f1_test,f1_test_svd))
if search:
    with open(save_path+txtname, 'a+', encoding='utf-8') as f:
        f.write(('test f1:{} -> {}'.format(f1_test,f1_test_svd)))